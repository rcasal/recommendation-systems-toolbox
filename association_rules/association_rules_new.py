#equired libraries are imported
import pandas as pd
import numpy as np
import mlxtend
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from google.cloud import bigquery

#SQL query is executed using the BigQuery client to extract transaction data from the database
query = """
SELECT 
visitor_visit_id,
array_agg(itemId) items_id,
array_agg(v2ProductName) items_name,
array_agg(v2ProductCategory) v2ProductCategory
from 
`mightyhive-data-science-poc.ds_ezemeli.recommender_vea_full_raw`
where rating = 5
group by visitor_visit_id
"""
client = bigquery.Client()
df = client.query(query).to_dataframe()
df.to_parquet("data/recommender_associetion_rules.parquet", index=False)
#metric: Metric to evaluate if a rule is of interest. supported metrics are 'support', 'confidence', 'lift'
#min_support: float (default: 0.5) The support is computed as the fraction transactions_where_item(s)_occur / total_transactions.
#min_threshold: float (default: 0.8) Minimal threshold for the evaluation metric, via the metric parameter, to decide whether a candidate rule is of interest.
# frequency_model: accepted apriori and eclat


class AssociationRules():
    def __init__(self, transactions_df: pd.DataFrame):
        self.transactions_df = transactions_df

        # Check if transactions_df data contains required columns
        if set(['visitor_visit_id', 'items_id', 'items_name', 'v2ProductCategory']).issubset(transactions_df.columns):
            transactions_df['transacction_len'] = transactions_df['items_id'].apply(lambda x: len(x))
            print("Cantidad media de productos por transaccion: ", int(transactions_df['transacction_len'].mean()))
            print("Transactions shape: ", transactions_df.shape)
        else:
            raise ValueError("Input DataFrame must contain columns 'visitor_visit_id', 'items_id', 'items_name' and 'v2ProductCategory'")
        
    def best_sellers(self, n_productos:int):
            # Aplanamos la lista de compras
            self.n_productos = n_productos
            total_purchases = []
            for i in self.transactions_df.items_name:
                for j in i:
                     total_purchases.append(j)
                    
            # Mostramos la lista de los Productos mas vendidos
            total_purchases  = pd.DataFrame( total_purchases , columns=['elementos'])
            self.best_sellers =  total_purchases.elementos.value_counts()[:self.n_productos]
            return self.best_sellers

    def fit(self,frequency_model: str,  min_support: float,  min_threshold: float,metric: str ='confidence', n_words: int=1) -> pd.DataFrame:
        # GENERAMOS LA LISTA DE PRODUCTOS TOTAL
        self.frequency_model = frequency_model.lower()
        self.metric = metric
        self.min_threshold = min_threshold
        self.min_support = min_support
        if (self.metric!='support') and (self.metric!='lift') and (self.metric!='confidence'):
            raise ValueError("Supported metrics are 'support', 'confidence', 'lift'")
        if (self.frequency_model!='apriori') and (self.frequency_model!='fpgrowth'):
            raise ValueError("Accepted frequency model 'apriori' or 'fpgrowth'")
        #Elegimos la cantidad de palabras del corpus (texto)
        product_id = []
        product_list_name = []
        product_category = []

        for i in range(self.transactions_df.items_id.shape[0]):
            for j,k,t in zip(self.transactions_df.items_id[i], self.transactions_df.items_name[i], self.transactions_df.v2ProductCategory[i]):
                
                # Naming convention needed (tomamos solo K palabras de cada producto)
                k = " ".join([x.lower() for x in k.split()[:n_words]])
                
                if j not in product_id and k not in product_list_name:
                    product_id.append(j)
                    product_list_name.append(k)
                    product_category.append(t)  
    
        
        products = pd.DataFrame(np.array([product_id, product_list_name, product_category]).T, columns=['item_id', 'item_name', 'item_category'])
        print("Hay ", products.shape[0], " productos.")


        #vectorizing

        # recorremos todas las transacciones
        transaction_matrix = []
        for i in range(self.transactions_df.items_id.shape[0]):
            #Creamos un vector Zeros con el largo de la cantidad de productos
            vector = np.zeros(len(products))
            for j in self.transactions_df.items_name[i]:
            
                #Aplicamos misma transformacion ("Naming Convention")
                j = " ".join([x.lower() for x in j.split()[:n_words]])
                try:
                # Si el producto se encuentra en la transaccion Ingresamos un 1 en el vector.
                    indice = product_list_name.index(j)
                    vector[indice] = 1
                except ValueError:
                    pass
            transaction_matrix.append(vector)
        #transformamos la lista en array
        transaction_matrix = np.array(transaction_matrix)
        #convertimos la matriz en df con el indice como lista de productos
        matrix_df = pd.DataFrame(data = transaction_matrix.T,index=product_id)

        matrix_df = matrix_df.T
        if (matrix_df.eq(0) | matrix_df.eq(1)).all().all():
            print("All elements are either 0 or 1")
        
        if self.frequency_model== 'apriori':
            frequent_itemsets = apriori(matrix_df, min_support=self.min_support, use_colnames=True)
            print("apriori completed")
        elif self.frequency_model== 'fpgrowth':
            frequent_itemsets = fpgrowth(matrix_df, min_support=self.min_support, use_colnames=True)
        
        if frequent_itemsets.empty:
            print("No frequent items for the ingested thersholds")
        else: 
            rules = association_rules(frequent_itemsets, metric=self.metric, min_threshold=self.min_threshold)
            rules['antecedents_no_frozen'] = rules['antecedents'].apply(lambda x: list(x)).astype("unicode")
            rules['consequents_no_frozen'] = rules['consequents'].apply(lambda x: list(x)).astype("unicode")

            return rules

    def recommend(antecedent, rules: pd.DataFrame , max_results: int = 6 ) -> pd.DataFrame:
        # get the rules for this antecedent
        """
        Select the DataFrame row that contains a column of frozen sets that is equal to a particular list converted to a frozen set.

        Args:
            df (pandas.DataFrame): The DataFrame to select from.
            frozenset_col (str): The name of the column containing the frozen sets.
            target_list (list): The list to convert to a frozen set and check for.

        Returns:
            pandas.DataFrame or None: The row that contains a column equal to the target frozen set, or None if the target frozen set was not found.
        """
        # Convert the target list to a frozen set
        target_frozenset = frozenset(antecedent)

        # Use apply() to check if any element in the frozen set column is equal to the target frozen set
        is_target_frozenset = rules['antecedents'].apply(lambda x: x == target_frozenset)
        
        # If any element in the frozen set column is equal to the target frozen set, return the corresponding row
        if is_target_frozenset.any():
            preds = rules.loc[is_target_frozenset]
            preds = preds.sort_values(by=['confidence',  'lift', 'support' , 'conviction'], ascending=False)[:max_results]
            return preds
        else:
            print("Antecedent not found in rules")
        
        
model = AssociationRules(df)
model.best_sellers(10)
rules = model.fit('fpgrowth', min_support=0.008,min_threshold=0.2)
pred = model.recommend(['7261', '9017', '308'], rules)
pred
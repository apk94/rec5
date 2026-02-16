import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

class Task(object):
    def __init__(self, file_name):
        self.data = pd.read_csv(file_name)
        self.data = self.data.drop('Amount', axis=1)
        self.data = self.data.drop('id', axis=1)
        for col in self.data.columns:
            if col.startswith('V'):
                self.data[col] = self.data[col].apply(lambda x: 0 if x < 0 else 1)
        self.data = self.data.astype(bool)
        self.frequent_itemsets = apriori(self.data, min_support=0.01, use_colnames=True)
        self.rules = association_rules(self.frequent_itemsets)


    def t1(self):
      frequent_itemsets2 = apriori(self.data, min_support=0.3, use_colnames=True)
      return frequent_itemsets2

    def t2(self):
      rules = association_rules(self.frequent_itemsets, metric="confidence", min_threshold=0.9)
      rules = rules[['antecedents', 'consequents', 'support', 'confidence']]
      return rules

    def t3(self):
      fraud_rules = self.rules[self.rules['consequents'] == frozenset({'Class'})]
      fraud_rules = fraud_rules[['antecedents', 'consequents', 'support', 'confidence']]
      return fraud_rules
        
    def t4(self):
     fraud_rules = self.t3()
     most_common = fraud_rules.sort_values('support', ascending=False).head(1)
     most_common = most_common[['antecedents', 'consequents', 'support', 'confidence']]
     return most_common

    def t5(self):
     fraud_rules = self.t3()
     most_consistent = fraud_rules.sort_values('confidence', ascending=False).head(1)
     most_consistent = most_consistent[['antecedents', 'consequents', 'support', 'confidence']]
     return most_consistent

if __name__ == "__main__":
    t = Task('creditcard_public.csv')
    print("----T1----" + "\n")
    print(str(t.t1()) + "\n")
    print("----T2----" + "\n")
    print(str(t.t2()) + "\n")
    print("----T3----" + "\n")
    print(str(t.t3()) + "\n")
    print("----T4----" + "\n")
    print(str(t.t4()) + "\n")
    print("----T5----" + "\n")
    print(str(t.t5()) + "\n")



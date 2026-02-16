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
        return None

    def t2(self):
        return None

    def t3(self):
        return None
        
    def t4(self):
        return None

    def t5(self):
        return None


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

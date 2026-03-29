import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score


data = {'sample': ['Hi dad, where are you?', 
                   'Free money now',
                   'Win a free cruise trip',
                   'Meeting at 10am',
                   'Click to claim your prize',
                   'Please send the meeting report',
                   'Assignment is due today!',
                   'Congratulations! You won!',
                   'Claim your $1000 prize',
                   'New Assignment Posted!'],
        
        'check': [0,1,1,0,1,0,1,0,0,1] }

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(df['sample'], df['check'], test_size=0.3, random_state=20)

num = TfidfVectorizer()
train = num.fit_transform(X_train)
test= num.transform(X_test)

m = MultinomialNB()
m.fit(train, y_train)

p = m.predict(test)
print(f"Accuracy: {accuracy_score(y_test, p) * 100}%")

x= input("Enter mail subject::")
inp=[x]
new_tfidf = num.transform(inp)
op = m.predict(new_tfidf)

for msg, pred in zip(inp, op):
    print(f"'{msg}' -> {'Spam' if pred == 1 else 'Not Spam'}")

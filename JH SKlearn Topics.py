import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

testdf = pd.read_csv('2000.csv', usecols=['Body', 'Publication Day Of Month', 'Publication Month', 'Publication Year'])

#Remove unnecessary symbols, numbers, words less than 3 characters and puts all text as lowercase
testdf['Body'].replace(to_replace='[,\.!?]', value='', inplace=True, regex=True)
testdf['Body'].replace(to_replace='\d+', value='', inplace=True, regex=True)
testdf['Body'].replace(to_replace=r'\b(\w{1,2})\b', value='', inplace=True, regex=True)
testdf['Body'] = testdf['Body'].str.lower()

#Generate Wordcloud
string_check = testdf['Body'].str.cat(sep=' ')

wordcloud = WordCloud(background_color='white', max_words=1000, contour_width=3, contour_color='steelblue')
wordcloud.generate(testdf['Body'].values[0])

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Generate TF Vectors
cv = CountVectorizer(stop_words='english')
test_cv = cv.fit_transform(testdf['Body'].head(40))
vocabulary = cv.get_feature_names()

doc_term_matrix = pd.DataFrame(test_cv.toarray(), columns=vocabulary)

lda = LatentDirichletAllocation(n_components=6)
lda.fit(test_cv)

for i, topic in enumerate(lda.components_):
    print('topic '+ str(i+1))
    print(' '.join([vocabulary[j] for j in topic.argsort()[:-16:-1]]))

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.stem import WordNetLemmatizer

def lemma_text(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in text]

main_df = pd.read_csv('2000.csv', usecols=['Body', 'Publication Day Of Month', 'Publication Month', 'Publication Year'])

#Remove NaN values, lowercase contents of Body column, filters for bush and gore and resets the index
print(main_df.shape)
main_df.dropna(subset=['Body'], inplace=True)
print(main_df.shape)

main_df['Body'] = main_df['Body'].str.lower()
main_df = main_df[main_df['Body'].str.contains('gore|bush')]
main_df = main_df.reset_index(drop=True)

#Create a single date column from day, month and year columns
main_df['Date'] = pd.to_datetime(main_df['Publication Year']*10000+main_df['Publication Month']*100+main_df['Publication Day Of Month'], format='%Y%m%d')
main_df.drop(['Publication Year','Publication Month', 'Publication Day Of Month'], axis=1, inplace=True)
print(main_df.shape)
print(main_df.head(10))

#Temporary check on first N documents
main_df = main_df[:500]

#Remove unnecessary symbols, numbers, words less than 3 characters and apply lemmatizer
main_df['Body'].replace(to_replace='[,\.!?]', value='', inplace=True, regex=True)
main_df['Body'].replace(to_replace='\d+', value='', inplace=True, regex=True)
main_df['Body'].replace(to_replace=r'\b(\w{1,2})\b', value='', inplace=True, regex=True)
main_df['Body'].apply(lemma_text)

#Generate Wordcloud
string_check = main_df['Body'].str.cat(sep=' ')

wordcloud = WordCloud(background_color='white', max_words=1000, contour_width=3, contour_color='steelblue')
wordcloud.generate(string_check)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Generate doc-term matrix
cv = CountVectorizer(stop_words='english')
ft_cv = cv.fit_transform(main_df['Body'])
vocabulary = cv.get_feature_names()

doc_term_matrix = pd.DataFrame(ft_cv.toarray(), columns=vocabulary)

#Fit LDA model to doc-term matrix
lda = LatentDirichletAllocation(n_components=6)
lda.fit(ft_cv)

print('log likelihood score: ' + str(lda.score(ft_cv)))

for i, topic in enumerate(lda.components_):
    print('topic '+ str(i))
    print(' '.join([vocabulary[j] for j in topic.argsort()[:-16:-1]]))

#Generate doc-topic matrix
lda_out = lda.transform(ft_cv)
doc_topic_matrix = pd.DataFrame(lda_out)
doc_topic_matrix['Date'] = main_df['Date']
print(doc_topic_matrix.head(100))
print(doc_topic_matrix.shape)
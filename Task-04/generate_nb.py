import nbformat as nbf

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("# Task 4: Sentiment Analysis on Social Media Data\n\nAnalyze and visualize sentiment patterns in social media data to understand public opinion and attitudes towards specific topics or brands."),
    
    nbf.v4.new_code_cell("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom wordcloud import WordCloud\nimport warnings\n\nwarnings.filterwarnings('ignore')\nplt.style.use('ggplot')"),
    
    nbf.v4.new_markdown_cell("## 1. Load the Dataset"),
    nbf.v4.new_code_cell("df = pd.read_csv('data-science-datasets-main/Task 4/twitter_training.csv', header=None)\ndf.columns = ['ID', 'Entity', 'Sentiment', 'Text']\ndf.head()"),
    
    nbf.v4.new_markdown_cell("## 2. Data Cleaning"),
    nbf.v4.new_code_cell("print('Original Shape:', df.shape)\n\n# Drop duplicates\ndf = df.drop_duplicates()\n\n# Drop missing values\ndf = df.dropna()\n\nprint('Shape after cleaning:', df.shape)"),
    
    nbf.v4.new_markdown_cell("## 3. Exploratory Data Analysis & Visualization"),
    nbf.v4.new_markdown_cell("### 3.1 Overall Sentiment Distribution"),
    nbf.v4.new_code_cell("plt.figure(figsize=(8,6))\nsns.countplot(data=df, x='Sentiment', palette='viridis', order=df['Sentiment'].value_counts().index)\nplt.title('Overall Sentiment Distribution')\nplt.xlabel('Sentiment')\nplt.ylabel('Count')\nplt.show()"),
    
    nbf.v4.new_markdown_cell("### 3.2 Top Entities by Tweet Volume"),
    nbf.v4.new_code_cell("top_entities = df['Entity'].value_counts().head(10).index\n\nplt.figure(figsize=(10,6))\nsns.countplot(data=df[df['Entity'].isin(top_entities)], y='Entity', order=top_entities, palette='Set2')\nplt.title('Top 10 Entities by Tweet Volume')\nplt.xlabel('Count')\nplt.ylabel('Entity')\nplt.show()"),
    
    nbf.v4.new_markdown_cell("### 3.3 Sentiment Distribution for Top Entities"),
    nbf.v4.new_code_cell("plt.figure(figsize=(14,8))\nsns.countplot(data=df[df['Entity'].isin(top_entities)], y='Entity', hue='Sentiment', order=top_entities, palette='muted')\nplt.title('Sentiment Distribution for Top 10 Entities')\nplt.xlabel('Count')\nplt.ylabel('Entity')\nplt.legend(title='Sentiment', loc='lower right')\nplt.show()"),
    
    nbf.v4.new_markdown_cell("### 3.4 Word Clouds for Sentiments"),
    nbf.v4.new_code_cell("def generate_wordcloud(sentiment, title):\n    text = ' '.join(str(tweet) for tweet in df[df['Sentiment'] == sentiment]['Text'])\n    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='inferno').generate(text)\n    plt.figure(figsize=(10,5))\n    plt.imshow(wordcloud, interpolation='bilinear')\n    plt.axis('off')\n    plt.title(title)\n    plt.show()\n\ngenerate_wordcloud('Positive', 'Word Cloud - Positive Tweets')\ngenerate_wordcloud('Negative', 'Word Cloud - Negative Tweets')"),
    
    nbf.v4.new_markdown_cell("## 4. Insights\n\n- The overall sentiment across all tweets shows a relatively balanced mix, with slightly more positive and negative tweets than neutral ones.\n- Brands and games like `MaddenNFL`, `LeagueOfLegends`, and `CallOfDuty` are among the most discussed entities.\n- The sentiment breakdown per entity reveals interesting brand-specific trends (e.g., some brands skew heavily negative while others skew positive).\n- The word clouds highlight the most common terms associated with positive vs. negative feelings, which can help in understanding specific pain points or features users love.")
]

with open('Task_04_Sentiment_Analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('Notebook created successfully.')

import re
import jieba
import jieba.analyse
import pandas as pd
import tqdm
# from gensim import corpora, similarities, models
# from gensim.models import LdaModel
# from gensim.models import CoherenceModel
# import matplotlib.pyplot as plt
import tomotopy as tp


# p_weibo_data = r'E:\Document\MyStrategy\DEDA_fdu24fall_finalproj_group1\weibo-search-master\weibo\结果文件' + '\\'
# p_lda_file = r'E:\Document\MyStrategy\DEDA_fdu24fall_finalproj_group1\weibo-public-opinion-analysis\LDA' + '\\'
# p_out = r'E:\Document\MyStrategy\DEDA_fdu24fall_finalproj_group1\weibo-public-opinion-analysis\output' + '\\'


def clean(line):
    """对一个文件的数据进行清洗"""
    rep = ['【】', '【', '】', '👍', '🤝',
           '🐮', '🙏', '🇨🇳', '👏', '❤️', '………', '🐰', '...、、', '，，', '..', '💪', '🤓',
           '⚕️', '👩', '🙃', '😇', '🍺', '🐂', '🙌🏻', '😂', '📖', '😭', '✧٩(ˊωˋ*)و✧', '🦐', '？？？？', '//', '😊', '💰', '😜', '😯',
           '(ღ˘⌣˘ღ)', '✧＼٩(눈౪눈)و/／✧', '🌎', '🍀', '🐴',
           '🌻', '🌱', '🌱', '🌻', '🙈', '(ง•̀_•́)ง！', '🉑️', '💩',
           '🐎', '⊙∀⊙！', '🙊', '【？', '+1', '😄', '🙁', '👇🏻', '📚', '🙇',
           '🙋', '！！！！', '🎉', '＼(^▽^)／', '👌', '🆒', '🏻',
           '🙉', '🎵', '🎈', '🎊', '0371-12345', '☕️', '🌞', '😳', '👻', '🐶', '👄', '\U0001f92e\U0001f92e', '😔', '＋1', '🛀', '🐸',
           '🐷', '➕1',
           '🌚', '：：', '💉', '√', 'x', '！！！', '🙅', '♂️', '💊', '👋', 'o(^o^)o', 'mei\u2006sha\u2006shi', '💉', '😪', '😱',
           '🤗', '关注', '……', '(((╹д╹;)))', '⚠️', 'Ծ‸Ծ', '⛽️', '😓', '🐵',
           '🙄️', '🌕', '…', '😋', '[]', '[', ']', '→_→', '💞', '😨', '&quot;', '😁', 'ฅ۶•ﻌ•♡', '😰', '🎙️',
           '🤧', '😫', '(ง•̀_•́)ง', '😁', '✊', '🚬', '😤', '👻', '😣', '：', '😷', '(*^▽^)/★*☆', '🐁', '🐔', '😘', '🍋', '(✪▽✪)',
           '(❁´ω`❁)', '1⃣3⃣', '(^_^)／', '☀️',
           '🎁', '😅', '🌹', '🏠', '→_→', '🙂', '✨', '❄️', '•', '🌤', '💓', '🔨', '👏', '😏', '⊙∀⊙！', '👍',
           '✌(̿▀̿\u2009̿Ĺ̯̿̿▀̿̿)✌',
           '😊', '👆', '💤', '😘', '😊', '😴', '😉', '🌟', '♡♪..𝙜𝙤𝙤𝙙𝙣𝙞𝙜𝙝𝙩•͈ᴗ•͈✩‧₊˚', '👪', '💰', '😎', '🍀', '🛍', '🖕🏼', '😂',
           '(✪▽✪)', '🍋', '🍅', '👀', '♂️', '🙋🏻', '✌️', '🥳', '￣￣)σ',
           '😒', '😉', '🦀', '💖', '✊', '💪', '🙄', '🎣', '🌾', '✔️', '😡', '😌', '🔥', '❤', '🏼', '🤭', '🌿', '丨', '✅', '🏥', 'ﾉ',
           '☀', '5⃣⏺1⃣0⃣', '🚣', '🎣', '🤯', '🌺',
           '🌸',
           ]
    pattern_0 = re.compile('#.*?#')  # 在用户名处匹配话题名称
    pattern_1 = re.compile('【.*?】')  # 在用户名处匹配话题名称
    pattern_2 = re.compile('肺炎@([\u4e00-\u9fa5\w\-]+)')  # 匹配@
    pattern_3 = re.compile('@([\u4e00-\u9fa5\w\-]+)')  # 匹配@
    # 肺炎@环球时报
    pattern_4 = re.compile(u'[\U00010000-\U0010ffff\uD800-\uDBFF\uDC00-\uDFFF]')  # 匹配表情
    pattern_5 = re.compile('(.*?)')  # 匹配一部分颜文字
    pattern_7 = re.compile('L.*?的微博视频')
    pattern_8 = re.compile('（.*?）')
    # pattern_9=re.compile(u"\|[\u4e00-\u9fa5]*\|")#匹配中文

    line = line.replace('O网页链接', '')
    line = line.replace('-----', '')
    line = line.replace('①', '')
    line = line.replace('②', '')
    line = line.replace('③', '')
    line = line.replace('④', '')
    line = line.replace('>>', '')
    line = re.sub(pattern_0, '', line, 0)  # 去除话题
    line = re.sub(pattern_1, '', line, 0)  # 去除【】
    line = re.sub(pattern_2, '', line, 0)  # 去除@
    line = re.sub(pattern_3, '', line, 0)  # 去除@
    line = re.sub(pattern_4, '', line, 0)  # 去除表情
    # line = re.sub(pattern_5, '', line, 0)  # 去除一部分颜文字
    line = re.sub(pattern_7, '', line, 0)
    line = re.sub(pattern_8, '', line, 0)
    line = re.sub(r'\[\S+\]', '', line, 0)  # 去除表情符号

    for i in rep:
        line = line.replace(i, '')
    return line


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords#停用词


def synwordslist(filepath):
    syn = dict()
    with open(filepath, 'r',encoding='utf-8') as f:
        for line in f:
            synword = line.strip().split("\t")
            num = len(synword)
            for i in range(1, num):
                syn[synword[i]] = synword[0]
    return syn  # 近义词典


def seg_sentence(sentence, stopwords, synwords):
    sentence = re.sub(u'[0-9\.]+', u'', sentence)

    sentence_seged =jieba.cut(sentence.strip(),cut_all=False,use_paddle=10)#默认精确模式
    output_list = []
    for word in sentence_seged:
        if word not in stopwords and word.__len__()>1 and word != '\t':
            if word in synwords.keys():#如果是同义词
                word = synwords[word]
            if word is not None:
                word.encode('utf-8')
                output_list.append(word)
    return output_list


def get_raw_data():
    p_in = r'D:/Desktop/grade 1/DEDA/HW/text analysis/text test/content data/欧洲航线.xlsx'
    data = pd.read_excel(p_in)
    return data


def split_words(row, stopwords, synwords):
    content = row['微博内容']
    sentence = clean(content)
    return seg_sentence(sentence, stopwords, synwords)

#
# def deal_lda(train, p_out):
#     #输入train，输出词典,texts和向量
#     id2word = corpora.Dictionary(train)     # Create Dictionary
#     texts = train                           # Create Corpus
#     corpus = [id2word.doc2bow(text) for text in texts]   # Term Document Frequency
#
#     #使用tfidf
#     tfidf = models.TfidfModel(corpus)
#     corpus = tfidf[corpus]
#
#     id2word.save(p_out+'deerwester.dict') #保存词典
#     corpora.MmCorpus.serialize(p_out+'deerwester.mm', corpus)#保存corpus
#
#     return id2word,texts,corpus
#
# def run(corpus_1,id2word_1,num,texts):
#     #标准LDA算法
#     lda_model = LdaModel(corpus=corpus_1,
#                          id2word=id2word_1,
#                         num_topics=num,
#                        passes=60,
#                        alpha=(50/num),
#                        eta=0.01,
#                        random_state=42)
#     # num_topics：主题数目
#     # passes：训练伦次
#     # num：每个主题下输出的term的数目
#     #输出主题
#     #topic_list = lda_model.print_topics()
#     #for topic in topic_list:
#         #print(topic)
#     # 困惑度
#     perplex=lda_model.log_perplexity(corpus_1)  # a measure of how good the model is. lower the better.
#     # 一致性
#     coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word_1, coherence='c_v')
#     coherence_lda = coherence_model_lda.get_coherence()
#     #print('\n一致性指数: ', coherence_lda)   # 越高越好
#     return lda_model,coherence_lda,perplex
#
#
# def compute_coherence_values(dictionary, corpus, texts,start, limit, step):
#     """
#     Compute c_v coherence for various number of topics
#
#     Parameters:
#     ----------
#     dictionary : Gensim dictionary
#     corpus : Gensim corpus
#     texts : List of input texts
#     limit : Max num of topics
#
#     Returns:
#     -------
#     model_list : List of LDA topic models
#     coherence_values : Coherence values corresponding to the LDA model with respective number of topics
#     """
#     coherence_values = []
#     perplexs=[]
#     model_list = []
#     for num_topic in range(start, limit, step):
#         #模型
#         lda_model,coherence_lda,perplex=run(corpus,dictionary,num_topic,texts)
#         #lda_model = LdaModel(corpus=corpus,num_topics=num_topic,id2word=dictionary,passes=50)
#         model_list.append(lda_model)
#         perplexs.append(perplex)#困惑度
#         #一致性
#         #coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
#         #coherence_lda = coherence_model_lda.get_coherence()
#         coherence_values.append(coherence_lda)
#
#     return model_list, coherence_values,perplexs
#
#
# def show_1(dictionary, corpus, texts, start, limit, step):
#     # 从 5 个主题到 30 个主题，步长为 5 逐次计算一致性，识别最佳主题数
#     model_list, coherence_values, perplexs = compute_coherence_values(dictionary, corpus, texts, start, limit, step)
#     # 输出一致性结果
#     n = 0
#     for m, cv in zip(perplexs, coherence_values):
#         print("主题模型序号数", n, "主题数目", (n + 4), "困惑度", round(m, 4), " 主题一致性", round(cv, 4))
#         n = n + 1
#     # 打印折线图
#     x = list(range(start, limit, step))
#     # 困惑度
#     plt.plot(x, perplexs)
#     plt.xlabel("Num Topics")
#     plt.ylabel("perplex  score")
#     plt.legend(("perplexs"), loc='best')
#     plt.show()
#     # 一致性
#     plt.plot(x, coherence_values)
#     plt.xlabel("Num Topics")
#     plt.ylabel("Coherence score")
#     plt.legend(("coherence_values"), loc='best')
#     plt.show()
#
#     return model_list
#

def find_k(docs, min_k, max_k, min_df):
    # min_df 词语最少出现在2个文档中
    import matplotlib.pyplot as plt
    scores = []
    for k in range(min_k, max_k):
        # seed随机种子，保证在大邓这里运行结果与你运行的结果一样
        mdl = tp.LDAModel(min_df=min_df, k=k, seed=555)
        for words in docs:
            if words:
                mdl.add_doc(words)
        mdl.train()
        coh = tp.coherence.Coherence(mdl)
        scores.append(coh.get_score())

    # x = list(range(min_k, max_k - 1))  # 区间最右侧的值。注意：不能大于max_k
    # print(x)
    # print()
    plt.plot(range(min_k, max_k), scores)
    plt.xlabel("number of topics")
    plt.ylabel("coherence")
    plt.show()




def main():
    raw_data = get_raw_data()
    jieba.load_userdict('D:/Desktop/grade 1/DEDA/HW/text analysis/text test/2_LDA/自建词表.txt')#加载自建词表
    stopwords = stopwordslist('D:/Desktop/grade 1/DEDA/HW/text analysis/text test/2_LDA/停用词表.txt')  # 这里加载停用词的路径
    synwords = synwordslist('D:/Desktop/grade 1/DEDA/HW/text analysis/text test/2_LDA/近义词表.txt')#这里加载近义词的路径
    sentence_split = []
    len(raw_data)
    for _, row in tqdm.tqdm(raw_data.iterrows()):
        if isinstance(row['微博内容'], str):
            sentence_split.append(split_words(row, stopwords, synwords))
    # id2word, texts, corpus = deal_lda(sentence_split, p_out)
    # lda_model,coherence_lda,perplex = run(corpus, id2word, 3, texts)
    # topic_list = lda.print_topics()
    # model_list = show_1(id2word, corpus, texts, 4, 16, 1)  # 找出困惑度和主题一致性最佳的，最好是不超过20个主题数,10个为宜
    # n = input('输入指定模型序号，以0为第一个: ')  # 还是需要手动，权衡比较
    # optimal_model = choose(model_list, int(n))
    pd.to_pickle(sentence_split, 'D:/Desktop/grade 1/DEDA/HW/text analysis/text test/content data/coastline_sentence_split.pkl')
    find_k(sentence_split, 2, 2, 2)

    import tomotopy as tp

    mdl = tp.LDAModel(k=5, min_df=2, seed=555)
    for words in sentence_split:
        if words:
            mdl.add_doc(words=words)
    mdl.train()
    for k in range(mdl.k):
        print('Top 10 words of topic #{}'.format(k))
        print(mdl.get_topic_words(k, top_n=10))
        print('\n')
    mdl.summary()


    import pyLDAvis
    import numpy as np

    #获取pyldavis需要的参数
    topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])
    doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
    doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
    doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
    vocab = list(mdl.used_vocabs)
    term_frequency = mdl.used_vocab_freq


    prepared_data = pyLDAvis.prepare(
        topic_term_dists,
        doc_topic_dists,
        doc_lengths,
        vocab,
        term_frequency,
        # start_index=0, # tomotopy话题id从0开始，pyLDAvis话题id从1开始
        sort_topics=False #注意：否则pyLDAvis与tomotopy内的话题无法一一对应。
    )


    #可视化结果存到html文件中
    pyLDAvis.save_html(prepared_data, 'D:/Desktop/grade 1/DEDA/HW/text analysis/text test/content data/coastline_ldavis.html')

    #notebook中显示
    pyLDAvis.display(prepared_data)






if __name__ == "__main__":
    main()
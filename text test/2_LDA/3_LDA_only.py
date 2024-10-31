# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:17:56 2024

@author: 30730
"""

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


# p_weibo_data = r'D:/Desktop/grade 1/DEDA/HW/text analysis/text test/content data' + '\\'
# p_lda_file = r'E:\Document\MyStrategy\DEDA_fdu24fall_finalproj_group1\weibo-public-opinion-analysis\LDA' + '\\'
# p_out = r'E:\Document\MyStrategy\DEDA_fdu24fall_finalproj_group1\weibo-public-opinion-analysis\output' + '\\'



def get_raw_data():
    p_in = r'text test/content data/红海危机.xlsx'
    data = pd.read_excel(p_in)
    return data


# def split_words(row, stopwords, synwords):
#     content = row['微博正文']
#     sentence = clean(content)
#     return seg_sentence(sentence, stopwords, synwords)

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


def stopwordslist():
    stopwords = [line.strip() for line in open('text test/2_LDA/停用词表.txt','r', encoding='utf-8').readlines()]
    return stopwords

def main():
    # raw_data = get_raw_data()
    # jieba.load_userdict('text test/2_LDA/自建词表.txt')#加载自建词表
    # stopwords = stopwordslist('text test/2_LDA/停用词表.txt')  # 这里加载停用词的路径
    # synwords = synwordslist(p_lda_file+'近义词表.txt')#这里加载近义词的路径
    # sentence_split = []
    # len(raw_data)
    # for _, row in tqdm.tqdm(raw_data.iterrows()):
    #     if isinstance(row['微博正文'], str):
    #         sentence_split.append(split_words(row, stopwords, synwords))
    # id2word, texts, corpus = deal_lda(sentence_split, p_out)
    # lda_model,coherence_lda,perplex = run(corpus, id2word, 3, texts)
    # topic_list = lda.print_topics()
    # model_list = show_1(id2word, corpus, texts, 4, 16, 1)  # 找出困惑度和主题一致性最佳的，最好是不超过20个主题数,10个为宜
    # n = input('输入指定模型序号，以0为第一个: ')  # 还是需要手动，权衡比较
    # optimal_model = choose(model_list, int(n))



    words_str = ""
    with open("text test/content data/cleaning_红海危机.txt",'r',encoding = 'utf-8') as f:
        for line in f:
            line = re.sub(u"[0-9\s+.!/,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）＞＜-]+", "", line)  # 去掉多余字符
            if line == "":continue
            line = line.replace("\n", "") # 去掉换行符
            seg_list = jieba.cut(line, cut_all=False)
            words_str += (" ".join(seg_list))
    stopwords = stopwordslist()
    sentenceslpit = [word for word in words_str.split(" ") if word not in stopwords and len(word) > 1] # 去除停用词和去除单个文字
    print(sentenceslpit)

    # pd.to_pickle(sentence_split, p_out+'sentence_split.pkl')
    find_k(sentenceslpit, 2, 2, 2)

    import tomotopy as tp

    mdl = tp.LDAModel(k=5, min_df=2, seed=555)
    for words in sentenceslpit:
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
    pyLDAvis.save_html(prepared_data, 'text test/content data/红海危机ldavis.html')

    #notebook中显示
    pyLDAvis.display(prepared_data)






if __name__ == "__main__":
    main()
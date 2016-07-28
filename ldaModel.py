#coding=utf-8

import jieba  # 结巴分词工具
import numpy as np
import lda

class WordSegemention:

    def __init__(self, sentence):
        self.sentence = sentence
        self.list_word = self.word_segment()

    def word_segment(self):

        """
        function:利用结巴分词工具分词
        :param sentence: 分词前的句原
        :return: list_word:分词后的词原
        """

        words = jieba.cut(self.sentence)
        list_word = list()
        for word in words:
            list_word.append(word)
        for word in list_word:
            if word.__len__() < 2:  # 去除标点符号
                list_word.remove(word)
        return list_word

    def get_words_with_segmention(self):
        return self.list_word


class LDAModel:

    def __init__(self, document_path):

        """
        :param filename:文件录入
        录入样例:多行评论
            ---------
            这东西很好
            烂透了
            ----------
        """
        self.sentence_with_segment, self.vocabulary = self.load_document(document_path)
        self.vsm_model = self.train_vsm_model()
        self.n_topics = 8  # 主题的个数
        self.words_in_topic = self.train_lda_model(self.n_topics)  # 获得lda主题模型下的词分布

    @staticmethod
    def load_document(document_path):
        """
        function:加载数据
        :param document_path: 加载文件的路径
        :return:句子分词后词原列表的comments_with_segment, 词汇表vocabulary
        """
        sentence_with_segment = list()
        vocabulary = list()
        f = open(document_path, 'r')
        sentences = f.readlines()
        for sentence in sentences:
            list_word = WordSegemention(sentence).get_words_with_segmention()
            sentence_with_segment.append(list_word)
            for word in list_word:
                if vocabulary.count(word) == 0:
                    vocabulary.append(word)
        return sentence_with_segment, vocabulary

    def train_vsm_model(self):
        """
        function: 将词汇表训练成VSM模型, 权重为TF
        :return:
        """
        vsm_model = list()
        for sentence in self.sentence_with_segment:
            vsm = [ i*0 for i in range(0, self.vocabulary.__len__(), 1)]
            for word in sentence:
                index = self.vocabulary.index(word)
                vsm[index] += 1
            vsm_model.append(vsm)
        vsm_model = np.array(vsm_model)
        return vsm_model

    def train_lda_model(self, n_topics):
        """
        function: 训练LDA模型
        :param: n_topic: 主题的个数
        :return: words_in_topic 主题内的词分布
        """
        model = lda.LDA(n_topics=n_topics, n_iter=200, random_state=1)
        model.fit(self.vsm_model)  # 填充vsm模型
        topic_word = model.topic_word_
        n_top_words = 8
        words_in_topic = dict()
        for i, topic_dict in enumerate(topic_word):
            topic_words = np.array(self.vocabulary)[np.argsort(topic_dict)][:-(n_top_words+1):-1]
            words_in_topic[i] = topic_words
        return words_in_topic

    def get_topics(self):
        return self.words_in_topic


def main():
    file_path = 'test.txt'
    lda = LDAModel(file_path)
    topics = lda.get_topics()
    for i in topics:
        print '------- topic %s ----------' % i
        for word in topics[i]:
            print word,
        print


if __name__ == '__main__':
    main()



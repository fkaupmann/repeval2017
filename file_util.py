from nltk import word_tokenize
import codecs
from  gensim.models import Word2Vec

def read_attr_adj_noun(filename,encoding='utf-8', vectorspace=False, verbosity = 0):
    """
    Read attribute-adjective-noun triples from the HeiPLAS files.
    :param filename:
    :param encoding:
    :param vectorspace:
    :param verbosity:
    :return:
    """
    result = []
    adjs = []
    nouns = []
    attrs = []
    with codecs.open(filename,encoding=encoding) as f:
        not_in_vec_space = []
        complete_string = f.read()
        # print complete_string
        tokens = word_tokenize(complete_string)
        # print tokens
        for i in range(0,len(tokens),3):

            aan = []
            for j in range(0,3):
                word = (tokens[i+j].lower())
                if word == 'direction_orientation':
                    word = 'direction'
                aan.append(word)
                if j == 0 and word not in attrs:
                    attrs.append(word)
                elif j == 1 and word not in adjs:
                    adjs.append(word)
                elif j == 2 and word not in nouns:
                    nouns.append(word)

            if vectorspace:
                try:
                    #falls vectorspace übergeben wurde, prüfe ob alle worte im space enthalten sind
                    attr_repr = vectorspace[aan[0].lower()]
                    adj_repr = vectorspace[aan[1].lower()]
                    noun_repr = vectorspace[aan[2].lower()]
                    result.append(aan)
                except KeyError:
                    not_in_vec_space.append(aan)
            else:
                result.append(aan)

    if verbosity >= 1:
        print("Es wurden {} Adj-Noun-Attribut-Phrasen eingelesen. {} enthielten Wörter, die nicht im Embedding-Space enthalten waren".format(len(result),len(not_in_vec_space)))

    return result, attrs, adjs, nouns

TEST_FILE_PATH = "/Users/Fabian/Documents/Uni/6. Semester/Bachelorarbeit/Data/mitchell-lapata-sim-ratings.txt"

def read_sim_ratings(filename, vectorspace=False, verbosity = 0):
    """Reads phrase pairs and similarity ratings from the dataset of mitchell and lapata (2010)."""
    result = []
    with codecs.open(filename, encoding='utf-8') as f:
        not_in_vec_space = []
        for line in f:
            line_list = line.split()
            if line_list[1] == 'adjectivenouns':
                if not vectorspace:
                    result.append(line_list[3:])
                else:
                    try:
                        adj1 = vectorspace[line_list[3]]
                        noun1 = vectorspace[line_list[4]]
                        adj2 = vectorspace[line_list[5]]
                        noun2 = vectorspace[line_list[6]]
                        result.append(line_list[3:])
                    except KeyError:
                        not_in_vec_space.append(line_list[3:])
    if verbosity >= 1:
        print("Es wurden {} Adj-Noun Phrasen-Paare eingelesen. {} enthielten Wörter, die nicht im Embedding-Space enthalten waren\nResultat: {}".format(len(result),len(not_in_vec_space),result))
    return result

def write_to_file(filename,string,encoding='utf-8'):
    """Writes string to certain file"""
    with codecs.open(filename,'a+',encoding='utf-8') as f:
        try:
            f.writelines(string + "\n")
        except UnicodeEncodeError:
            print("Encoding Error: %s" % string)
        except UnicodeDecodeError:
            print("Decoding Error: %s" % string)

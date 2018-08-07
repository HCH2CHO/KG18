
"""
fr = open("../data/movie2/entity2id.txt")
fr_1 = open("../data/movie2/entity_embedding.txt")
fw = open("../data/movie2/results.txt",'w')

temp = ""
for line in fr.readlines():
    entity_name = line.split("\t")[0]
    embedding_data = fr_1.readline()
    temp = entity_name + "\t" + embedding_data + "\n"
    fw.write(temp)

fr.close()
fr_1.close()
fw.close()
"""

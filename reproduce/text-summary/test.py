from Extract_Summary_English import Extract_Summary_English

f = open('./text3.txt', encoding='utf-8')
data = f.readlines()
f.close()
print(data)
text = ""
for line in data:
    text += line

ese = Extract_Summary_English(text)
result = ese.ExtractSummary()
print("result")
print(result)
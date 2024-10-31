
#只读取微博内容并保存
import pandas as pd

def excel_one_line_to_list():
    df = pd.read_excel(r'text test/content data/红海危机.xlsx', usecols=[4])  # 读取微博内容
    df_li = df.values.tolist()
    result = []
    for s_li in df_li:
        result.append(s_li[0])
    print(result)
    df = pd.DataFrame(result, columns=['微博内容'])
    df.to_excel("text test/output/内容_红海危机.xlsx", index=False)
if __name__ == '__main__':
    excel_one_line_to_list()



#将微博内容excel转txt

df = pd.read_excel(r'text test/output/内容_红海危机.xlsx')		# 使用pandas模块读取数据

print('开始写入txt文件...')
df.to_csv('text test/output/内容_红海危机.txt', header=None, sep=',', index=False)		# 写入，逗号分隔
print('文件写入成功!')

    
    

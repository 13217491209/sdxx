import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息
from sklearn.model_selection import StratifiedKFold  # 用于交叉验证
from sklearn.metrics import roc_auc_score  # 用于计算ROC-AUC评分
from sklearn.model_selection import train_test_split  # 用于数据划分（训练集和测试集）
from catboost import CatBoostClassifier  # 引入CatBoost分类器
from sklearn.preprocessing import LabelEncoder  # 引入标签编码器

# 读取数据
train = pd.read_csv("train.csv")  # 读取训练数据
test = pd.read_csv("test.csv")  # 读取测试数据
sub = pd.read_csv("submission.csv")  # 读取提交样例（用于存储预测结果）

# 合并训练集和测试集，方便进行数据处理
data = pd.concat([train, test])

# 将'incident_date'列转换为日期时间格式
data['incident_date'] = pd.to_datetime(data['incident_date'], format='%Y-%m-%d')

# 设置一个起始日期，计算每条记录到该日期的天数
startdate = datetime.datetime.strptime('2022-06-30', '%Y-%m-%d')
data['time'] = data['incident_date'].apply(lambda x: startdate - x).dt.days

# 对数据中的分类变量进行标签编码（Label Encoding）
numerical_fea = list(data.select_dtypes(include=['object']).columns)  # 获取所有的对象类型的列名（即分类特征）
division_le = LabelEncoder()  # 初始化LabelEncoder对象
for fea in numerical_fea:
    division_le.fit(data[fea].values)  # 拟合每个特征的分类标签
    data[fea] = division_le.transform(data[fea].values)  # 将每个分类特征转换为数字标签
print("数据预处理完成!")  # 打印提示信息

# 将数据分成训练集和测试集
testA = data[data['fraud'].isnull()].drop(['policy_id', 'incident_date', 'fraud'], axis=1)  # 测试集数据（fraud为缺失值的部分）
trainA = data[data['fraud'].notnull()]  # 训练集数据（fraud有值的部分）

# 获取训练集的特征和标签
data_x = trainA.drop(['policy_id', 'incident_date', 'fraud'], axis=1)  # 特征
data_y = train[['fraud']].copy()  # 标签（fraud列）

# 需要处理的列（需要在模型训练中作为类别特征处理）
col = ['policy_state', 'insured_sex', 'insured_education_level', 'incident_type', 'collision_type',
       'incident_severity', 'authorities_contacted', 'incident_state', 'incident_city',
       'police_report_available', 'auto_make', 'auto_model']

# 对需要处理的类别特征进行转换为字符串类型，以便CatBoost处理
for i in data_x.columns:
    if i in col:
        data_x[i] = data_x[i].astype('str')  # 转换为字符串类型
for i in testA.columns:
    if i in col:
        testA[i] = testA[i].astype('str')  # 转换为字符串类型

# 初始化CatBoostClassifier模型
model = CatBoostClassifier(
    loss_function="Logloss",  # 损失函数为Logloss
    eval_metric="AUC",  # 评估指标为AUC
    task_type="CPU",  # 使用CPU进行训练
    learning_rate=0.1,  # 学习率
    iterations=10000,  # 最大迭代次数
    random_seed=2020,  # 随机种子，保证结果可复现
    od_type="Iter",  # 使用迭代类型的早期停止
    depth=7,  # 树的深度
    early_stopping_rounds=300  # 如果300轮内没有提升，则停止训练
)

# 交叉验证
answers = []  # 存储每一折交叉验证的预测结果
mean_score = 0  # 初始化AUC的平均值
n_folds = 10  # 设置交叉验证的折数为10
sk = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2019)  # 初始化StratifiedKFold，用于分层K折交叉验证
for train, test in sk.split(data_x, data_y):
    # 获取训练集和测试集
    x_train = data_x.iloc[train]
    y_train = data_y.iloc[train]
    x_test = data_x.iloc[test]
    y_test = data_y.iloc[test]

    # 在训练集上训练模型，并在验证集上评估
    clf = model.fit(x_train, y_train, eval_set=(x_test, y_test), verbose=500, cat_features=col)

    # 预测验证集，并计算AUC
    yy_pred_valid = clf.predict(x_test)
    print('cat验证的auc:{}'.format(roc_auc_score(y_test, yy_pred_valid)))  # 输出验证集的AUC值

    # 累加每一折的AUC得分，计算平均AUC
    mean_score += roc_auc_score(y_test, yy_pred_valid) / n_folds

    # 对测试集数据进行预测，返回预测的概率值
    y_pred_valid = clf.predict(testA, prediction_type='Probability')[:, -1]
    answers.append(y_pred_valid)  # 存储每一折的预测结果

# 输出10折交叉验证的平均AUC
print('10折平均AUC:{}'.format(mean_score))

# 将每一折的预测结果取平均值作为最终的预测结果
lgb_pre = sum(answers) / n_folds

# 将最终预测结果保存到提交文件中
sub['fraud'] = lgb_pre
sub.to_csv('预测.csv', index=False)  # 保存预测结果为CSV文件，去掉索引

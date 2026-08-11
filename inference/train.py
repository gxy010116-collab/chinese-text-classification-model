#!/usr/bin/env python3
"""
Fine-tune hfl/chinese-roberta-wwm-ext for 10-class Chinese text classification.

Generates a training dataset programmatically (no external data dependency),
fine-tunes on CPU or GPU, and saves the checkpoint to inference/checkpoints/.

Usage:
    python train.py                      # default: 400 samples/class, 3 epochs
    python train.py --epochs 5           # more epochs
    python train.py --samples 200        # fewer training samples (faster on CPU)
    python train.py --device cpu         # force CPU
    python train.py --seed 42            # fixed random seed for reproducibility
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup

# ---------------------------------------------------------------------------
# Labels (must match classifier.py)
# ---------------------------------------------------------------------------
LABEL_MAP: List[str] = [
    "财经", "科技", "教育", "体育", "娱乐",
    "时政", "社会", "房产", "健康", "军事",
]
NUM_LABELS: int = len(LABEL_MAP)

HERE: Path = Path(__file__).resolve().parent
CHECKPOINT_DIR: Path = HERE / "checkpoints"

log = logging.getLogger(__name__)


# ===================================================================
# Training data generation
# ===================================================================

# ---- domain-specific keyword pools ----
_KEYWORDS: Dict[int, List[str]] = {
    0: [  # 财经
        "央行", "A股", "沪指", "深成指", "创业板", "科创板", "北向资金", "南向资金",
        "存款准备金率", "基准利率", "降息", "加息", "量化宽松", "通胀", "CPI", "PPI",
        "人民币汇率", "离岸人民币", "在岸人民币", "美元指数", "外汇储备", "外汇占款",
        "GDP", "固定资产投资", "社会消费品零售总额", "进出口", "贸易顺差", "贸易逆差",
        "减税降费", "增值税", "企业所得税", "个人所得税", "税收优惠", "财政赤字",
        "专项债", "特别国债", "地方债", "城投债", "信用债", "可转债",
        "IPO", "定增", "配股", "并购重组", "私有化", "借壳上市",
        "国有大行", "股份制银行", "城商行", "农商行", "券商", "公募基金", "私募基金",
        "数字货币", "区块链金融", "移动支付", "第三方支付", "金融科技", "互联网金融",
        "上市公司年报", "季报", "业绩预告", "利润分配", "分红", "回购",
        "国际油价", "黄金价格", "铜价", "铁矿石", "大宗商品", "期货市场",
        "标普500", "纳斯达克", "道琼斯", "恒生指数", "日经225", "富时100",
        "世界银行", "国际货币基金组织", "亚投行", "金砖银行", "丝路基金",
    ],
    1: [  # 科技
        "芯片", "半导体", "光刻机", "晶圆", "晶体管", "纳米制程", "EDA", "RISC-V",
        "人工智能", "深度学习", "神经网络", "大模型", "GPT", "LLM", "AIGC", "计算机视觉",
        "5G", "6G", "基站", "频谱", "毫米波", "物联网", "边缘计算", "数字孪生",
        "鸿蒙", "安卓", "iOS", "操作系统", "开源", "Github", "开发者", "API",
        "云计算", "云原生", "容器", "K8s", "微服务", "Serverless", "SaaS", "PaaS",
        "自动驾驶", "L4", "L5", "激光雷达", "毫米波雷达", "车路协同", "智能座舱",
        "卫星", "运载火箭", "载人航天", "空间站", "月球探测", "火星探测", "北斗导航",
        "特斯拉", "苹果", "华为", "小米", "OPPO", "vivo", "比亚迪", "蔚来", "小鹏",
        "量子计算", "超导", "量子比特", "量子纠错", "量子通信", "量子密钥分发",
        "元宇宙", "VR", "AR", "XR", "空间计算", "脑机接口", "可穿戴设备",
        "腾讯", "阿里", "字节跳动", "百度", "京东", "美团", "拼多多", "快手",
        "机器人", "人形机器人", "工业机器人", "服务机器人", "具身智能", "灵巧手",
        "碳中和", "光伏", "风电", "储能", "锂电池", "钠电池", "固态电池", "氢能",
    ],
    2: [  # 教育
        "高考", "中考", "考研", "考公", "考编", "四六级", "托福", "雅思", "GRE",
        "清华", "北大", "复旦", "交大", "浙大", "南大", "中科大", "哈工大",
        "教育部", "教育厅", "教育局", "教委", "学位办", "考试院", "招生办",
        "双减", "课后服务", "延时服务", "五项管理", "作业管理", "考试管理",
        "新课标", "核心素养", "学科核心素养", "大单元教学", "项目式学习", "跨学科",
        "职业教育", "产教融合", "校企合作", "学徒制", "双师型", "1+X证书",
        "在线教育", "慕课", "微课", "翻转课堂", "智慧课堂", "AI助教",
        "出国留学", "海外留学", "留学回国", "公派留学", "自费留学", "留学预警",
        "学区房", "派位", "划片", "就近入学", "多校划片", "公民同招",
        "强基计划", "综合评价", "自主招生", "保送生", "特长生", "艺术类招生",
        "学前教育", "幼儿园", "普惠园", "托育", "早教", "幼小衔接",
        "教育信息化", "智慧教育", "教育新基建", "国家智慧教育平台", "数字校园",
        "大学生就业", "校招", "管培生", "应届生", "春招", "秋招",
    ],
    3: [  # 体育
        "奥运会", "亚运会", "全运会", "世锦赛", "世界杯", "亚洲杯", "欧冠", "NBA", "CBA",
        "男足", "女足", "男篮", "女篮", "男排", "女排", "国乒", "国羽", "体操队", "游泳队",
        "梅西", "C罗", "姆巴佩", "哈兰德", "詹姆斯", "库里", "杜兰特",
        "姚明", "刘翔", "苏炳添", "谷爱凌", "郑钦文", "朱婷", "孙颖莎",
        "马拉松", "半马", "越野跑", "铁人三项", "自行车赛", "F1", "摩托车赛",
        "中超", "中甲", "中乙", "英超", "西甲", "意甲", "德甲", "法甲",
        "网球大满贯", "澳网", "法网", "温网", "美网", "中网", "上海大师赛",
        "电竞", "英雄联盟", "王者荣耀", "DOTA2", "CS2", "瓦洛兰特", "亚运电竞",
        "滑雪", "花样滑冰", "短道速滑", "速度滑冰", "冰球", "冰壶", "自由式滑雪",
        "世界杯预选赛", "世预赛", "40强赛", "12强赛", "18强赛", "附加赛",
        "转会", "签约", "续约", "解约", "租借", "自由身", "转会窗口",
        "体彩", "竞彩", "胜负彩", "进球彩", "半全场", "比分",
    ],
    4: [  # 娱乐
        "电影", "电视剧", "网剧", "综艺", "纪录片", "短视频", "微短剧", "互动剧",
        "春节档", "暑期档", "国庆档", "贺岁档", "五一档", "端午档", "中秋档",
        "票房", "排片率", "上座率", "场均人次", "口碑", "豆瓣评分", "猫眼评分",
        "奥斯卡", "金球奖", "戛纳", "柏林", "威尼斯", "金鸡奖", "百花奖", "华表奖",
        "周杰伦", "陈奕迅", "林俊杰", "五月天", "张学友", "王菲", "邓紫棋",
        "流量明星", "顶流", "塌房", "人设", "热搜", "八卦", "爆料", "实锤",
        "B站", "抖音", "快手", "小红书", "微博", "知乎", "虎扑", "豆瓣",
        "米哈游", "原神", "崩坏", "绝区零", "网易游戏", "腾讯游戏", "鹰角",
        "steam", "epic", "3A大作", "独立游戏", "手游", "端游", "主机游戏",
        "音乐节", "演唱会", "LiveHouse", "音乐剧", "话剧", "脱口秀", "相声",
        "爱奇艺", "腾讯视频", "优酷", "芒果TV", "B站大会员", "Netflix", "Disney+",
        "AI作曲", "AI绘画", "AI编剧", "虚拟偶像", "数字人", "Vtuber",
    ],
    5: [  # 时政
        "国务院", "全国人大", "全国政协", "中央政治局", "中央委员会", "中央纪委",
        "国家主席", "国务院总理", "全国人大常委会委员长", "全国政协主席",
        "外交部", "国防部", "发改委", "财政部", "商务部", "工信部", "科技部",
        "生态环境部", "农业农村部", "人社部", "公安部", "司法部", "民政部",
        "一带一路", "人类命运共同体", "全球发展倡议", "全球安全倡议", "全球文明倡议",
        "中美关系", "中俄关系", "中欧关系", "中日关系", "中韩关系", "中印关系",
        "南海", "东海", "台海", "钓鱼岛", "黄岩岛", "仁爱礁", "九段线",
        "RCEP", "CPTPP", "DEPA", "IPEF", "APEC", "G20", "金砖", "上合组织",
        "改革开放", "中国式现代化", "共同富裕", "高质量发展", "新质生产力",
        "供给侧结构性改革", "需求侧管理", "扩大内需", "统一大市场",
        "双碳", "能耗双控", "碳排放权交易", "绿色金融", "ESG", "可持续发展",
        "法治中国", "法治政府", "依法行政", "司法改革", "监察体制改革",
        "两岸关系", "一国两制", "粤港澳大湾区", "横琴", "前海", "南沙",
    ],
    6: [  # 社会
        "地震", "台风", "暴雨", "暴雪", "洪涝", "干旱", "山体滑坡", "泥石流",
        "春运", "暑运", "小长假", "黄金周", "高速免费", "12306", "候补购票",
        "老龄化", "少子化", "人口普查", "出生率", "结婚率", "离婚率", "生育率",
        "志愿者", "义工", "献血", "慈善", "捐款", "众筹", "水滴筹", "轻松筹",
        "垃圾分类", "限塑令", "禁塑令", "光盘行动", "低碳生活", "绿色出行",
        "乡村振兴", "脱贫攻坚", "驻村帮扶", "对口支援", "东西部协作", "易地搬迁",
        "安全生产", "矿难", "火灾", "爆炸", "塌方", "透水", "瓦斯突出",
        "见义勇为", "好人好事", "感动中国", "最美", "榜样", "时代楷模",
        "食品安全", "315", "消费者权益", "价格欺诈", "虚假宣传", "退一赔三",
        "流浪动物", "野生动物保护", "自然保护区", "国家公园", "生物多样性",
        "雄安新区", "长三角一体化", "京津冀协同", "成渝双城", "中部崛起",
        "网络暴力", "网络谣言", "人肉搜索", "网暴治理", "清朗行动",
    ],
    7: [  # 房产
        "房价", "均价", "成交价", "挂牌价", "指导价", "评估价", "楼面价", "溢价率",
        "商品房", "商品住房", "商住两用", "住宅", "别墅", "洋房", "小高层", "高层",
        "首付", "首套", "二套", "房贷利率", "LPR", "公积金贷款", "组合贷款",
        "限购", "限售", "限价", "限贷", "放开限购", "取消限售", "松绑",
        "万科", "碧桂园", "恒大", "融创", "保利", "华润置地", "龙湖", "绿城",
        "土拍", "招拍挂", "溢价", "流拍", "底价成交", "熔断", "摇号",
        "二手房", "存量房", "次新房", "老破小", "学区", "地铁房", "江景房",
        "公摊面积", "套内面积", "得房率", "容积率", "绿化率", "车位比",
        "物业费", "业委会", "物管", "维修基金", "公维金", "电梯改造",
        "保障房", "公租房", "廉租房", "共有产权房", "人才房", "安置房",
        "住建部", "住建厅", "房管局", "不动产登记", "网签", "备案", "过户",
        "城市更新", "旧改", "棚改", "拆迁", "货币化安置", "实物安置",
    ],
    8: [  # 健康
        "医院", "三甲", "二甲", "社区医院", "乡镇卫生院", "村卫生室", "诊所",
        "医保", "医保目录", "医保谈判", "带量采购", "集采", "DRG", "DIP",
        "新冠", "疫苗", "mRNA", "灭活", "重组蛋白", "腺病毒载体", "加强针",
        "流感", "肺炎", "乙肝", "艾滋病", "结核病", "疟疾", "登革热", "猴痘",
        "高血压", "糖尿病", "冠心病", "脑卒中", "慢阻肺", "哮喘", "慢性肾病",
        "癌症", "肿瘤", "靶向药", "免疫治疗", "CAR-T", "PD-1", "化疗", "放疗",
        "中医药", "中药", "针灸", "推拿", "拔罐", "艾灸", "汤剂", "丸剂", "颗粒剂",
        "手术", "微创", "腔镜", "达芬奇机器人", "器官移植", "人工心脏", "人工关节",
        "心理健康", "抑郁症", "焦虑症", "失眠", "心理咨询", "心理治疗", "正念",
        "体检", "筛查", "早筛", "早诊", "早治", "癌症筛查", "基因检测",
        "卫健委", "疾控中心", "药监局", "医保局", "中医药管理局",
        "近视", "肥胖", "脊柱侧弯", "龋齿", "学生常见病", "学校卫生",
        "养生", "保健", "食疗", "药膳", "八段锦", "太极拳", "广场舞",
    ],
    9: [  # 军事
        "航母", "驱逐舰", "护卫舰", "潜艇", "两栖攻击舰", "补给舰", "扫雷舰",
        "歼-20", "歼-35", "歼-16", "歼-15", "歼-10", "轰-6", "运-20", "直-20",
        "东风", "巨浪", "长剑", "鹰击", "霹雳", "红旗", "海红旗",
        "火箭军", "战略导弹", "洲际导弹", "高超音速", "滑翔弹头", "分导式多弹头",
        "南海舰队", "东海舰队", "北海舰队", "航母编队", "远洋训练", "护航编队",
        "五角大楼", "北约", "美日同盟", "美韩同盟", "美菲同盟", "AUKUS", "QUAD",
        "台海", "第一岛链", "第二岛链", "反介入", "区域拒止", "A2/AD",
        "军演", "实弹演习", "联合演习", "多国演习", "环太平洋", "金色眼镜蛇",
        "国防白皮书", "国防预算", "军费", "国防开支", "军力报告", "军控",
        "无人机", "察打一体", "蜂群", "忠诚僚机", "巡飞弹", "反无人机",
        "电子战", "网络战", "太空战", "信息战", "认知战", "混合战争",
        "核武器", "核弹头", "核威慑", "核裁军", "不扩散", "禁核试",
    ],
}

_TEMPLATES: Dict[int, List[str]] = {
    0: [  # 财经
        "{kw}发布最新数据表明经济持续向好",
        "分析人士指出{kw}走势将对市场产生深远影响",
        "{kw}板块今日表现强势 多只个股涨幅超5%",
        "专家解读{kw}最新政策：利好实体经济",
        "{kw}再创新高 市场情绪明显回暖",
        "监管层就{kw}发布新规 旨在防范金融风险",
        "{kw}指数单日暴涨 投资者信心提振",
        "报告显示{kw}同比增长 经济复苏态势明朗",
        "{kw}利率调整 对居民储蓄影响几何",
        "{kw}开放新举措 外资机构积极布局",
        "三部门联合发文规范{kw} 促进行业健康发展",
        "{kw}成市场焦点 机构密集调研相关企业",
        "央行就{kw}表态：保持政策连续性稳定性",
        "{kw}和{kw}联动 市场预期改善",
        "各地出台措施支持{kw}发展 稳增长意图明显",
    ],
    1: [  # 科技
        "{kw}技术取得重大突破 引发行业关注",
        "{kw}发布新一代产品 性能提升显著",
        "专家解读{kw}发展趋势：前景广阔",
        "{kw}赛道融资火热 多家创业公司获投",
        "研究报告显示{kw}市场规模将超千亿",
        "{kw}与{kw}加速融合 催生新业态",
        "国内{kw}企业加速出海 竞争优势凸显",
        "{kw}标准制定取得进展 国际话语权提升",
        "最新{kw}专利获批 技术壁垒进一步巩固",
        "全球{kw}峰会召开 中国方案受关注",
        "{kw}产业化进程加快 成本持续下降",
        "科技部支持{kw}研发 设立专项基金",
        "{kw}领域的中国创新引发国际关注",
        "业内热议{kw}落地场景 商业化前景可期",
        "{kw}团队获国际大奖 展现中国科研实力",
    ],
    2: [  # 教育
        "教育部就{kw}发布最新通知 引发广泛讨论",
        "{kw}改革持续推进 多地出台实施细则",
        "专家解读{kw}政策：促进教育公平",
        "{kw}话题再上热搜 家长群体高度关注",
        "调查显示{kw}成效显著 获得社会认可",
        "2025年{kw}安排出炉 这些变化要知道",
        "多地推进{kw}试点 探索教育改革新路径",
        "高校{kw}改革 培养模式与时俱进",
        "{kw}和{kw}如何平衡 专家给出建议",
        "中小学生{kw}现状调查：喜忧参半",
        "两会代表委员热议{kw} 提出多项建议",
        "新形势下{kw}面临挑战与机遇",
        "数字化助力{kw} 教育信息化提速",
        "国际比较视野下的{kw} 中国表现亮眼",
        "{kw}领域的优秀实践获教育部推广",
    ],
    3: [  # 体育
        "{kw}最新战报 精彩对决引爆球迷热情",
        "{kw}创造历史 中国体育迎来新突破",
        "{kw}赛季正式启动 赛程安排出炉",
        "中国{kw}选手在国际赛场再创佳绩",
        "{kw}联赛战况激烈 冠军归属悬念重重",
        "{kw}名将宣布重要决定 引起广泛关注",
        "{kw}创纪录 刷新历史最好成绩",
        "{kw}赛事门票开售 球迷热情高涨",
        "分析：{kw}为何能脱颖而出",
        "{kw}决赛精彩回顾 经典瞬间令人难忘",
        "新赛季{kw}规则调整 对比赛有何影响",
        "{kw}和{kw}上演巅峰对决 观众大饱眼福",
        "中国{kw}队公布大名单 新面孔引关注",
        "{kw}发展迎来新机遇 政策支持力度加大",
        "青训体系助力{kw} 后备人才不断涌现",
    ],
    4: [  # 娱乐
        "{kw}热映引发观影热潮 观众口碑持续发酵",
        "{kw}收视率再创新高 话题讨论量破亿",
        "{kw}发布新作品 艺术风格获好评",
        "深度解析{kw}现象：为何如此火爆",
        "{kw}和{kw}梦幻联动 引发粉丝狂欢",
        "{kw}获国际大奖 中国文娱走向世界",
        "{kw}上线即爆 播放量24小时破千万",
        "最新{kw}综艺开播 阵容豪华备受期待",
        "{kw}创作团队接受专访 讲述幕后故事",
        "行业观察：{kw}赛道竞争白热化",
        "{kw}即将上线 预约人数突破500万",
        "乐评人评{kw}新专辑：诚意之作",
        "{kw}网播量夺冠 登顶热度榜",
        "跨年晚会{kw}压轴 收视峰值破5%",
        "{kw}IP开发加速 衍生周边销售火爆",
    ],
    5: [  # 时政
        "{kw}召开重要会议 部署当前重点工作",
        "外交部就{kw}答记者问 阐明中方立场",
        "{kw}发表声明 对事件表示严重关切",
        "国务院常务会议审议通过{kw}方案",
        "{kw}和{kw}举行会谈 达成多项共识",
        "全国人大就{kw}进行立法调研",
        "中央经济工作会议强调{kw}的重要性",
        "{kw}白皮书发布 全面阐述政策主张",
        "中方就{kw}问题向有关方面提出交涉",
        "多部门联合印发{kw}指导意见",
        "{kw}入选年度十大新闻",
        "政治局会议分析研究{kw}形势",
        "{kw}国际合作取得新进展",
        "中央深改委审议通过{kw}改革方案",
        "国务院印发{kw}通知 明确实施路径",
    ],
    6: [  # 社会
        "{kw}最新进展 救援工作持续进行中",
        "{kw}引发关注 相关部门已介入调查",
        "{kw}数据公布 折射社会发展新变化",
        "{kw}创下新纪录 网友纷纷点赞",
        "多地出现{kw} 相关部门发布预警",
        "{kw}情况通报 善后工作有序开展",
        "暖心！{kw}中的感人瞬间",
        "{kw}整治初见成效 市民纷纷叫好",
        "{kw}现象引发社会讨论 专家这样看",
        "关注{kw} 这些措施即将落地",
        "{kw}和{kw}联动 社会治理创新实践",
        "报告显示{kw}改善明显 群众获得感增强",
        "紧急通知：{kw}来袭 请做好防范",
        "这些关于{kw}的新规下月起实施",
        "直击{kw}现场 亲历者讲述经过",
    ],
    7: [  # 房产
        "{kw}最新数据出炉 趋势如何解读",
        "多地调整{kw}政策 市场反应积极",
        "{kw}成交量回暖 业内怎么看",
        "分析人士解读{kw}走势：稳字当头",
        "{kw}迎来新变化 购房者需关注",
        "一线城市{kw}市场观察：分化加剧",
        "住建部回应{kw}问题：因城施策",
        "{kw}再出新政 意在提振市场信心",
        "{kw}和{kw}双双上涨 释放什么信号",
        "最新{kw}数据公布 行业迎来转机",
        "{kw}新规征求意见 这些要点值得关注",
        "业内热议{kw}：拐点是否已至",
        "多地{kw}去化周期缩短 库存压力缓解",
        "{kw}开发投资数据发布 下行趋势趋缓",
        "专家支招{kw}：刚需购房者可关注这些区域",
    ],
    8: [  # 健康
        "国家卫健委就{kw}发布最新指南",
        "研究表明{kw}与{kw}存在关联 引起医学界关注",
        "{kw}疫苗研发取得进展 有望年内获批",
        "专家提醒{kw}高发季注意做好防护",
        "新版{kw}诊疗指南发布 新增多项内容",
        "{kw}早期筛查率提升 早诊早治效果显著",
        "药监局批准{kw}新药上市 惠及众多患者",
        "关注{kw} 多部门联合开展专项行动",
        "新型{kw}治疗技术临床试验取得积极结果",
        "各地加强{kw}防控 保障群众健康",
        "最新{kw}数据公布 防控成效显著",
        "中医治疗{kw}取得新突破 疗效获认可",
        "关于{kw}的几大误区 专家逐一澄清",
        "{kw}纳入医保 减轻患者负担",
        "世界卫生组织就{kw}发布全球行动倡议",
    ],
    9: [  # 军事
        "{kw}正式服役 国防实力再上新台阶",
        "中国{kw}在演习中表现出色 获高度评价",
        "{kw}最新动态 外媒密集报道",
        "国防部就{kw}答记者问 回应外界关切",
        "{kw}技术突破 填补国内空白",
        "中国{kw}赴海外参加联合演习",
        "{kw}发展白皮书发布 透明化程度提升",
        "{kw}和{kw}协同演练 体系作战能力增强",
        "专家分析{kw}发展趋势：自主创新是关键",
        "新型{kw}亮相 引发军事观察家热议",
        "{kw}实战化训练常态化 战备水平提升",
        "中国{kw}部队现代化建设加速推进",
        "{kw}在国际防务展上展出 多国表示兴趣",
        "深度解析{kw}性能：达到世界先进水平",
        "{kw}领域国际合作不断深化",
    ],
}


def _generate_samples_for_label(
    label_id: int,
    num_samples: int,
    rng: random.Random,
) -> List[Dict[str, object]]:
    """Generate diverse training samples for a single label."""
    keywords: List[str] = _KEYWORDS[label_id]
    templates: List[str] = _TEMPLATES[label_id]
    samples: List[Dict[str, object]] = []
    seen: set = set()

    while len(samples) < num_samples:
        tmpl: str = rng.choice(templates)
        # pick 1-2 keywords for this sample
        used: List[str] = [rng.choice(keywords)]
        if "{kw}" in tmpl and tmpl.count("{kw}") >= 2:
            # need a second distinct keyword
            k2: str = rng.choice(keywords)
            if k2 != used[0]:
                used.append(k2)
            else:
                k2 = rng.choice([k for k in keywords if k != used[0]])
                used.append(k2)

        text: str = tmpl
        for kw in used:
            text = text.replace("{kw}", kw, 1)

        # deduplicate
        if text in seen:
            continue
        seen.add(text)

        samples.append({"text": text, "label": label_id})

        if len(seen) >= len(templates) * len(keywords):
            break  # exhausted combinations

    return samples


def generate_training_data(
    samples_per_class: int = 400,
    seed: int = 42,
) -> List[Dict[str, object]]:
    """Generate a complete training dataset with balanced class distribution."""
    rng = random.Random(seed)
    all_samples: List[Dict[str, object]] = []

    for lid in range(NUM_LABELS):
        samples = _generate_samples_for_label(lid, samples_per_class, rng)
        all_samples.extend(samples)
        log.info(
            "  Label %d (%s): %d samples generated",
            lid,
            LABEL_MAP[lid],
            len(samples),
        )

    rng.shuffle(all_samples)
    return all_samples


# ===================================================================
# Dataset
# ===================================================================


class TextClassificationDataset(Dataset):
    """Torch Dataset wrapping a list of {"text": ..., "label": ...} dicts."""

    def __init__(
        self,
        samples: List[Dict[str, object]],
        tokenizer: BertTokenizer,
        max_length: int = 128,
    ):
        self.texts: List[str] = [str(s["text"]) for s in samples]
        self.labels: List[int] = [int(s["label"]) for s in samples]  # type: ignore[arg-type]
        self.tokenizer: BertTokenizer = tokenizer
        self.max_length: int = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ===================================================================
# Model (matches classifier.py TextClassifier)
# ===================================================================


class TextClassifier(nn.Module):
    """BERT-base backbone + Linear classification head."""

    def __init__(
        self,
        model_name: str = "hfl/chinese-roberta-wwm-ext",
        num_labels: int = NUM_LABELS,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.bert: BertModel = BertModel.from_pretrained(model_name)
        self.dropout: nn.Dropout = nn.Dropout(dropout_rate)
        self.classifier: nn.Linear = nn.Linear(
            self.bert.config.hidden_size, num_labels
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_hidden: torch.Tensor = outputs.last_hidden_state[:, 0, :]
        pooled: torch.Tensor = self.dropout(cls_hidden)
        logits: torch.Tensor = self.classifier(pooled)

        result: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)
        return result


# ===================================================================
# Training
# ===================================================================


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    return float((preds == labels).float().mean().item())


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss: float = 0.0
    total_acc: float = 0.0
    n_batches: int = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs["loss"].item()
            total_acc += compute_accuracy(outputs["logits"], batch["labels"])
            n_batches += 1
    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def train(
    model_name: str = "hfl/chinese-roberta-wwm-ext",
    samples_per_class: int = 400,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_length: int = 128,
    device_str: Optional[str] = None,
    seed: int = 42,
    val_split: float = 0.15,
) -> None:
    # ---- seed everything ----
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---- device ----
    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info("Device: %s", device)

    # ---- generate training data ----
    log.info("Generating training data (%d samples/class)...", samples_per_class)
    all_samples: List[Dict[str, object]] = generate_training_data(
        samples_per_class=samples_per_class, seed=seed
    )
    rng = random.Random(seed)
    rng.shuffle(all_samples)

    split_idx: int = int(len(all_samples) * (1.0 - val_split))
    train_samples: List[Dict[str, object]] = all_samples[:split_idx]
    val_samples: List[Dict[str, object]] = all_samples[split_idx:]
    log.info(
        "Total: %d  Train: %d  Val: %d",
        len(all_samples),
        len(train_samples),
        len(val_samples),
    )

    # ---- tokenizer ----
    log.info("Loading tokenizer: %s", model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # ---- datasets & dataloaders ----
    train_ds = TextClassificationDataset(train_samples, tokenizer, max_length)
    val_ds = TextClassificationDataset(val_samples, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ---- model ----
    log.info("Loading backbone: %s", model_name)
    model = TextClassifier(model_name=model_name, num_labels=NUM_LABELS)
    model.to(device)

    # ---- optimizer & scheduler ----
    total_steps: int = len(train_loader) * epochs
    warmup_steps: int = int(total_steps * warmup_ratio)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ---- training loop ----
    best_val_acc: float = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss: float = 0.0
        epoch_acc: float = 0.0
        n_batches: int = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss: torch.Tensor = outputs["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()

            acc: float = compute_accuracy(outputs["logits"], batch["labels"])
            epoch_loss += loss.item()
            epoch_acc += acc
            n_batches += 1
            pbar.set_postfix(
                loss=f"{loss.item():.3f}", acc=f"{acc:.3f}"
            )

        avg_train_loss: float = epoch_loss / max(n_batches, 1)
        avg_train_acc: float = epoch_acc / max(n_batches, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)

        log.info(
            "Epoch %d - train_loss=%.4f train_acc=%.4f  val_loss=%.4f val_acc=%.4f",
            epoch,
            avg_train_loss,
            avg_train_acc,
            val_loss,
            val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_DIR / "pytorch_model.bin")
            meta = {
                "model_name": model_name,
                "num_labels": NUM_LABELS,
                "label_map": LABEL_MAP,
                "seed": seed,
                "samples_per_class": samples_per_class,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "max_length": max_length,
                "best_val_accuracy": round(best_val_acc, 4),
            }
            with open(CHECKPOINT_DIR / "training_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            log.info("  -> Best checkpoint saved (val_acc=%.4f)", best_val_acc)

    log.info("Training complete. Best val_acc: %.4f", best_val_acc)


# ===================================================================
# CLI
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune Chinese text classification model"
    )
    parser.add_argument(
        "--model-name",
        default="hfl/chinese-roberta-wwm-ext",
        help="HuggingFace model ID (default: hfl/chinese-roberta-wwm-ext)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=400,
        help="Training samples per class (default: 400, total ~4000)",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--max-length", type=int, default=128, help="Max token length (default: 128)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Force device (cpu, cuda, mps)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )

    t0: float = time.perf_counter()

    train(
        model_name=args.model_name,
        samples_per_class=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        device_str=args.device,
        seed=args.seed,
    )

    elapsed: float = time.perf_counter() - t0
    log.info("Total training time: %.1f min", elapsed / 60)


if __name__ == "__main__":
    main()

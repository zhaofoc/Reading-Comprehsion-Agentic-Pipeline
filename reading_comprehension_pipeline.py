"""
基于OpenAI GPT-4o的Multi-Agent阅读理解生成系统
各Agent使用独立的LLM与不同的temperature
包含五种不同类型的问题生成器
添加了Justification Agent用于生成错误选项的解释
"""

from typing import List, Dict, Any
import random
from openai import OpenAI
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
import csv


class GrammarAspectSelector:
    """词汇选择器 - 根据正确率动态调整欧标等级"""
    
    def __init__(self, csv_path: str = "word.csv"):
        """
        初始化选择器
        Args:
            csv_path: word.csv 文件路径，假设第3列（索引2）是欧标等级
        """
        self.words_by_level = {
            'A1': [], 'A2': [], 'B1': [], 'B2': [], 'C1': [], 'C2': []
        }
        
        # 读取CSV文件
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过表头
                for row in reader:
                    if len(row) >= 3:
                        word = row[0]  # 第1列是单词
                        level = row[2].strip().upper()  # 第3列是欧标等级
                        if level in self.words_by_level:
                            self.words_by_level[level].append(word)
        except FileNotFoundError:
            print(f"[Warning] 未找到 {csv_path} 文件，使用示例词汇")
            self._init_sample_words()
        
        # 欧标等级顺序
        self.levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
        self.current_level = 'A1'  # 初始等级
    
    def _init_sample_words(self):
        """初始化示例词汇"""
        self.words_by_level = {
            'A1': ['hello', 'world', 'good', 'bad', 'happy', 'sad', 'cat', 'dog', 'book', 'school'],
            'A2': ['important', 'difficult', 'beautiful', 'interesting', 'family', 'friend', 'music', 'sport', 'food', 'travel'],
            'B1': ['technology', 'environment', 'education', 'culture', 'experience', 'challenge', 'opportunity', 'communication', 'develop', 'achieve'],
            'B2': ['sophisticated', 'innovation', 'sustainability', 'perspective', 'analysis', 'comprehensive', 'significant', 'contemporary', 'flexibility', 'efficiency'],
            'C1': ['phenomenal', 'articulate', 'profound', 'nuanced', 'paradigm', 'meticulous', 'ambiguous', 'deliberate', 'constitute', 'emanate'],
            'C2': ['ephemeral', 'quintessential', 'perspicacious', 'obfuscate', 'ameliorate', 'egregious', 'ubiquitous', 'pellucid', 'serendipity', 'mellifluous']
        }
    
    def select_words(self, accuracy: float = None) -> List[str]:
        """
        根据正确率选择10个单词
        Args:
            accuracy: 上一轮的正确率 (0-1之间)
        Returns:
            选中的10个单词列表
        """
        # 根据正确率调整等级
        if accuracy is not None:
            if accuracy > 0.8:
                self._increase_level()
            elif accuracy < 0.5:
                self._decrease_level()
        
        # 获取当前等级的词汇
        level_words = self.words_by_level[self.current_level]
        
        # 如果该等级词汇不足10个，使用所有可用词汇
        if len(level_words) < 10:
            print(f"[Warning] {self.current_level} 等级词汇不足10个，使用所有 {len(level_words)} 个")
            selected = level_words[:]
        else:
            # 随机选择10个
            selected = random.sample(level_words, 10)
        
        print(f"\n[Grammar Aspect Selector] 当前等级: {self.current_level}")
        print(f"[Grammar Aspect Selector] 选择的词汇: {', '.join(selected)}")
        
        return selected
    
    def _increase_level(self):
        """提升一个等级"""
        current_index = self.levels.index(self.current_level)
        if current_index < len(self.levels) - 1:
            self.current_level = self.levels[current_index + 1]
            print(f"[Grammar Aspect Selector] 正确率 > 80%，等级提升至: {self.current_level}")
    
    def _decrease_level(self):
        """降低一个等级"""
        current_index = self.levels.index(self.current_level)
        if current_index > 0:
            self.current_level = self.levels[current_index - 1]
            print(f"[Grammar Aspect Selector] 正确率 < 50%，等级降低至: {self.current_level}")


class ParagraphGenerator:
    """文章生成器"""

    def __init__(self, llm):
        self.llm = llm
        self.generated_topics = []

    def generate(self, words: List[str]) -> str:
        previous_topics_str = ""
        if self.generated_topics:
            previous_topics_str = f"\n已使用过的主题（请避免）：{', '.join(self.generated_topics)}"

        prompt = f"""请基于以下词汇生成一篇适合阅读理解的英文文章（200-300词）：
{', '.join(words)}

要求：
1. 文章需要有明确的主题和结构
2. 适合中高级英语学习者
3. 包含一些给定的词汇
4. 内容要有深度，适合出阅读理解题
{previous_topics_str}

请直接输出文章内容。"""

        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        article = response.content.strip()

        topic_identifier = ' '.join(article.split()[:20])
        self.generated_topics.append(topic_identifier)

        print(f"\n[Paragraph Generator] 生成文章完成，长度: {len(article.split())} 词")
        return article


class ParagraphValidator:
    """文章验证器"""

    def __init__(self, llm):
        self.llm = llm

    def validate(self, article: str) -> str:
        prompt = f"""请检查以下英文文章的正确性，包括语法、逻辑、用词等方面：

{article}

如果发现错误，请修正后返回完整的正确文章。
如果文章没有问题，请直接返回原文。
"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        print(f"\n[Paragraph Validator] 文章验证完成")
        return response.content.strip()


class BaseQuestionGenerator:
    """问题生成器基类"""

    def __init__(self, llm):
        self.llm = llm

    def _parse_questions(self, text: str) -> List[Dict[str, Any]]:
        questions = []
        lines = text.split("\n")
        current = {}

        for line in lines:
            line = line.strip()
            if line.startswith("Question"):
                if current:
                    questions.append(current)
                current = {"question": line.split(":", 1)[1].strip() if ":" in line else "", "options": {}}
            elif len(line) >= 2 and line[0] in ("A", "B", "C", "D") and line[1] == ")":
                current["options"][line[0]] = line[2:].strip()
            elif line.startswith("Correct Answer"):
                answer_part = line.split(":", 1)[1].strip() if ":" in line else ""
                # 提取单个字母（A/B/C/D）
                current["answer"] = answer_part[0] if answer_part else ""

        if current and current.get("question"):
            questions.append(current)

        return questions

    def generate(self, article: str) -> List[Dict[str, Any]]:
        raise NotImplementedError


class GistQuestionGenerator(BaseQuestionGenerator):
    """主旨题生成器 - General Meaning类的问题"""

    def generate(self, article: str) -> List[Dict[str, Any]]:
        prompt = f"""基于以下文章，生成1道主旨理解题（考察文章的主要思想）：

{article}

要求：
1. 题目应该问"文章的主要思想是什么？"或类似的主旨问题
2. 4个选项（A、B、C、D）
3. 选项应该涵盖不同的理解角度，但只有一个能准确概括文章主旨
4. 干扰项应该是文章中提到但不是主旨的内容，或者是常见的误解
5. 使用英文出题

请按以下格式输出：
Question 1: [题目]
A) [选项A]
B) [选项B]
C) [选项C]
D) [选项D]
Correct Answer: [A/B/C/D]
"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        questions_text = response.content.strip()
        questions = self._parse_questions(questions_text)

        print(f"\n[Gist Question Generator] 生成了 {len(questions)} 道主旨理解题")
        return questions


class SpecificQuestionGenerator(BaseQuestionGenerator):
    """细节题生成器 - Particular Meanings类的问题"""

    def generate(self, article: str) -> List[Dict[str, Any]]:
        prompt = f"""基于以下文章，生成1道细节理解题（考察文章中的具体事实和细节信息）：

{article}

要求：
1. 题目应该询问具体的事实、数据、时间、地点等细节信息
2. 4个选项（A、B、C、D）
3. 正确答案必须能在文章中直接找到
4. 干扰项应该与正确答案相似，但有细微差别，考察学生的细心程度
5. 使用英文出题

示例题目类型：
- "When does...?"
- "Where is...?"
- "How many...?"
- "Which of the following did...?"

请按以下格式输出：
Question 1: [题目]
A) [选项A]
B) [选项B]
C) [选项C]
D) [选项D]
Correct Answer: [A/B/C/D]
"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        questions_text = response.content.strip()
        questions = self._parse_questions(questions_text)

        print(f"\n[Specific Question Generator] 生成了 {len(questions)} 道细节理解题")
        return questions


class InferenceQuestionGenerator(BaseQuestionGenerator):
    """推理题生成器 - Reading Between the Lines类的问题"""

    def generate(self, article: str) -> List[Dict[str, Any]]:
        prompt = f"""基于以下文章，生成1道推理题（考察学生的推理和逻辑能力）：

{article}

要求：
1. 题目应该要求学生根据文章信息进行推理（"What can be inferred...?"，"What does...suggest?"等）
2. 正确答案不能在文章中直接出现，但能从文章逻辑推出
3. 4个选项（A、B、C、D）
4. 干扰项应该是文中明确提到的信息，或者是看起来合理但无法推出的内容
5. 使用英文出题

示例题目类型：
- "What can be inferred about...?"
- "The author suggests that..."
- "It can be concluded that..."
- "What does...imply?"

请按以下格式输出：
Question 1: [题目]
A) [选项A]
B) [选项B]
C) [选项C]
D) [选项D]
Correct Answer: [A/B/C/D]
"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        questions_text = response.content.strip()
        questions = self._parse_questions(questions_text)

        print(f"\n[Inference Question Generator] 生成了 {len(questions)} 道推理题")
        return questions


class AbsenceQuestionGenerator(BaseQuestionGenerator):
    """否定题生成器 - Identifying what is NOT mentioned"""

    def generate(self, article: str) -> List[Dict[str, Any]]:
        prompt = f"""基于以下文章，生成1道否定题（要求学生识别文章中未提及的内容）：

{article}

要求：
1. 题目使用否定形式：如"Which of the following is NOT mentioned...?"，"Which is NOT...?"等
2. 三个选项应该在文章中明确提到
3. 一个选项在文章中没有被提及或与文章内容矛盾
4. 选项应该都在相同的主题范围内，避免过于明显的干扰项
5. 使用英文出题

示例题目类型：
- "Which of the following is NOT mentioned as...?"
- "All of the following are true EXCEPT..."
- "The passage does NOT mention..."
- "Which is NOT mentioned in the text?"

请按以下格式输出：
Question 1: [题目]
A) [选项A - 文中提及]
B) [选项B - 文中提及]
C) [选项C - 文中提及]
D) [选项D - 文中未提及]
Correct Answer: [A/B/C/D]
"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        questions_text = response.content.strip()
        questions = self._parse_questions(questions_text)

        print(f"\n[Absence Question Generator] 生成了 {len(questions)} 道否定题")
        return questions


class LexicalQuestionGenerator(BaseQuestionGenerator):
    """词汇题生成器 - Meaning of words in context"""

    def generate(self, article: str) -> List[Dict[str, Any]]:
        prompt = f"""基于以下文章，生成1道词汇理解题（考察学生根据上下文理解单词含义的能力）：

{article}

要求：
1. 选择一个在文章中出现过的关键词汇（最好是中级或以上难度的单词）
2. 题目格式：询问这个单词在文章中的含义，而不是字典定义
3. 4个选项（A、B、C、D）
4. 正确答案应该是这个单词在特定上下文中的含义
5. 干扰项可以是：该单词的字典定义、其他常见含义、或文章中其他词汇的含义
6. 使用英文出题

示例题目类型：
- "In the passage, the word '...' most nearly means..."
- "The word '...' in the text refers to..."
- "As used in the passage, '...' suggests..."
- "Which of the following best defines '...' as it is used in the text?"

请按以下格式输出：
Question 1: [题目及文中词汇]
A) [选项A]
B) [选项B]
C) [选项C]
D) [选项D]
Correct Answer: [A/B/C/D]
"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        questions_text = response.content.strip()
        questions = self._parse_questions(questions_text)

        print(f"\n[Lexical Question Generator] 生成了 {len(questions)} 道词汇理解题")
        return questions


class EnhancedQuestionGenerator:
    """增强型题目生成器 - 集成五种不同类型的问题生成器"""

    def __init__(self, llm):
        self.llm = llm
        self.gist_generator = GistQuestionGenerator(llm)
        self.specific_generator = SpecificQuestionGenerator(llm)
        self.inference_generator = InferenceQuestionGenerator(llm)
        self.absence_generator = AbsenceQuestionGenerator(llm)
        self.lexical_generator = LexicalQuestionGenerator(llm)

    def generate(self, article: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        为一篇文章生成五种不同类型的问题
        
        Returns:
            包含五种问题类型的字典
        """
        print("\n[Enhanced Question Generator] 开始生成五种类型的问题...")
        
        all_questions = {
            "gist": self.gist_generator.generate(article),
            "specific": self.specific_generator.generate(article),
            "inference": self.inference_generator.generate(article),
            "absence": self.absence_generator.generate(article),
            "lexical": self.lexical_generator.generate(article),
        }

        total_questions = sum(len(q) for q in all_questions.values())
        print(f"\n[Enhanced Question Generator] 共生成 {total_questions} 道题目（5种类型）")
        
        return all_questions


class DistractorValidator:
    """干扰项验证器"""

    def __init__(self, llm):
        self.llm = llm

    def validate(self, article: str, questions_dict: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        验证所有类型的问题
        
        Args:
            article: 文章内容
            questions_dict: 包含五种类型问题的字典
            
        Returns:
            验证后的问题字典
        """
        valid_questions_dict = {}
        
        for question_type, questions in questions_dict.items():
            valid_questions = []
            
            for i, q in enumerate(questions, 1):
                prompt = f"""基于以下文章，请验证题目的正确性：

文章：
{article}

题目类型: {question_type}
{q['question']}
A) {q['options'].get('A', 'N/A')}
B) {q['options'].get('B', 'N/A')}
C) {q['options'].get('C', 'N/A')}
D) {q['options'].get('D', 'N/A')}
标注答案: {q.get('answer', 'N/A')}

请回答：这道题是否合格？回答"合格"或"不合格"，并说明理由。"""

                response = self.llm.invoke([HumanMessage(content=prompt)])
                result = response.content.strip()

                if "合格" in result and "不合格" not in result:
                    print(f"\n[Distractor Validator] [{question_type}] 题目 {i} 验证通过")
                    valid_questions.append(q)
                else:
                    print(f"\n[Distractor Validator] [{question_type}] 题目 {i} 未通过 → 丢弃")
                    print(f"理由: {result[:100]}...")
            
            valid_questions_dict[question_type] = valid_questions
        
        return valid_questions_dict


class JustificationAgent:
    """解释生成器 - 为错误选项生成解释"""

    def __init__(self, llm):
        self.llm = llm

    def generate_justifications(self, article: str, questions_dict: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        为所有错误选项生成解释
        
        Args:
            article: 文章内容
            questions_dict: 包含五种类型问题的字典
            
        Returns:
            添加了解释的问题字典
        """
        justified_questions_dict = {}
        
        for question_type, questions in questions_dict.items():
            justified_questions = []
            
            for i, q in enumerate(questions, 1):
                # 为每个选项生成解释
                justifications = {}
                correct_answer = q.get('answer', 'A')
                
                for option_key in ['A', 'B', 'C', 'D']:
                    if option_key not in q.get('options', {}):
                        continue
                    
                    # 跳过正确答案，不生成解释
                    if option_key == correct_answer:
                        justifications[option_key] = "[CORRECT]"
                        continue
                    
                    option_text = q['options'][option_key]
                    
                    prompt = f"""基于以下文章和题目，请为错误选项生成一段3-4句的英文解释，说明为什么这个选项是错误的。

文章：
{article}

题目类型: {question_type}
题目: {q['question']}
正确答案: {correct_answer}) {q['options'].get(correct_answer, 'N/A')}

要解释的错误选项:
{option_key}) {option_text}

要求：
1. 用英文编写
2. 3-4句话
3. 清晰解释这个选项为什么不符合文章内容或题目要求
4. 如果适用，引用文章中的相关信息
5. 语言要学生友好，易于理解

直接输出解释，不需要前缀。"""

                    response = self.llm.invoke([HumanMessage(content=prompt)])
                    justifications[option_key] = response.content.strip()
                
                # 将解释添加到问题中
                q['justifications'] = justifications
                justified_questions.append(q)
                
                print(f"\n[Justification Agent] [{question_type}] 题目 {i} 的解释已生成")
            
            justified_questions_dict[question_type] = justified_questions
        
        return justified_questions_dict


class StudentLearningUIWithJustification:
    """增强版学生学习界面 - 显示解释"""
    
    def display_and_evaluate(self, article: str, questions_dict: Dict[str, List[Dict[str, Any]]]) -> float:
        """
        展示文章、题目和解释，模拟学生评估
        
        Returns:
            平均正确率
        """
        print("\n" + "=" * 80)
        print("【阅读文章】\n")
        print(article)

        total_correct = 0
        total_questions = 0

        for question_type, questions in questions_dict.items():
            if not questions:
                continue
                
            print(f"\n【{question_type.upper()}类型题目】")
            
            for idx, q in enumerate(questions, 1):
                total_questions += 1
                print(f"\n{idx}. {q.get('question', 'N/A')}")
                
                for k in ['A', 'B', 'C', 'D']:
                    if k in q.get('options', {}):
                        print(f"   {k}) {q['options'][k]}")
                
                correct_answer = q.get('answer', 'N/A')
                print(f"   正确答案: {correct_answer}")
                
                # 显示解释
                justifications = q.get('justifications', {})
                if justifications:
                    print(f"\n   【选项解释】")
                    for opt_key in ['A', 'B', 'C', 'D']:
                        if opt_key in justifications:
                            if opt_key == correct_answer:
                                print(f"   {opt_key}) ✓ [正确答案]")
                            else:
                                print(f"   {opt_key}) ✗ {justifications[opt_key]}")
                
                # 模拟学生答题
                total_correct += 1

        accuracy = total_correct / total_questions if total_questions > 0 else 0
        
        print("\n" + "=" * 80)
        print(f"本轮正确率: {accuracy:.1%} ({total_correct}/{total_questions})")
        print("=" * 80)
        
        return accuracy


class MultiAgentReadingSystem:
    """多智能体阅读理解系统"""

    def __init__(self, api_key: str):
        """
        初始化系统
        
        Args:
            api_key: OpenAI API密钥
        """
        # 创建不同温度的LLM实例
        self.pg_llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.7)
        self.qg_llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.5)
        self.low_temp_llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.1)

        # 初始化各个Agent
        self.grammar_selector = GrammarAspectSelector()
        self.paragraph_generator = ParagraphGenerator(self.pg_llm)
        self.paragraph_validator = ParagraphValidator(self.low_temp_llm)
        self.question_generator = EnhancedQuestionGenerator(self.qg_llm)
        self.distractor_validator = DistractorValidator(self.low_temp_llm)
        self.justification_agent = JustificationAgent(self.low_temp_llm)
        self.student_ui = StudentLearningUIWithJustification()

    def run_cycle(self, cycle_num: int, accuracy: float = None):
        """
        运行一个完整的循环
        
        Args:
            cycle_num: 循环编号
            accuracy: 上一轮的正确率
            
        Returns:
            本轮的正确率
        """
        print(f"\n{'#' * 80}")
        print(f"第 {cycle_num} 轮循环")
        print(f"{'#' * 80}")

        # 第1步：选择词汇
        words = self.grammar_selector.select_words(accuracy)
        
        # 第2步：生成文章
        article = self.paragraph_generator.generate(words)
        
        # 第3步：验证文章
        validated_article = self.paragraph_validator.validate(article)
        
        # 第4步：生成五种类型的题目
        questions_dict = self.question_generator.generate(validated_article)
        
        # 第5步：验证题目
        valid_questions_dict = self.distractor_validator.validate(validated_article, questions_dict)

        # 第6步：生成解释
        if any(valid_questions_dict.values()):
            valid_questions_dict = self.justification_agent.generate_justifications(validated_article, valid_questions_dict)

        # 第7步：展示给学生并评估
        if any(valid_questions_dict.values()):
            accuracy = self.student_ui.display_and_evaluate(validated_article, valid_questions_dict)
        else:
            print("\n[Warning] 没有通过验证的题目，本轮正确率默认为 0.9")
            accuracy = 0.9

        return accuracy

    def run(self, cycles: int = 3):
        """
        运行完整系统
        
        Args:
            cycles: 循环次数
        """
        accuracy = None
        
        for i in range(1, cycles + 1):
            accuracy = self.run_cycle(i, accuracy)

        print("\n" + "=" * 80)
        print("所有循环完成！")
        print("=" * 80)


def main():
    # 从环境变量或直接设置API密钥
    import os
    API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    
    if API_KEY == "your-api-key-here":
        print("[Error] 请设置OPENAI_API_KEY环境变量或在代码中填入API密钥")
        return
    
    system = MultiAgentReadingSystem(API_KEY)
    system.run(cycles=3)


if __name__ == "__main__":
    main()

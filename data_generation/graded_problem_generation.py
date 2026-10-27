from utils import find_math_answer, is_equal
from jupyter_use import JupyterNotebookKernel
import os
import json
import time
import google.generativeai as genai


od = os.path.dirname
os.chdir(od(od(__file__)))
class ResBulid():  # 传递可用于修改的res.text
    pass

def load_jsonl(path: str):
    if not os.path.exists(path):
        with open(path, "w", encoding='utf-8') as f:
            pass
    with open(path, "r", encoding='utf-8') as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def get_problem_and_answer(data_point):
    return data_point["messages"][1]["content"][0]["content"], data_point["messages"][2]["content"][-1]["content"]

def get_lce_chain_from_data(d):
    # sys_prompt = "Below is a math problem. Please solve it step by step.\n"
    # problem = "Math Problem: " + d["messages"][1]["content"][0]["content"] + "\n"
    lce_chain = "<|assistant|>\n"
    for unit_m in d["messages"][2]["content"]:
        lce_chain += "<|"+unit_m["type"]+"|>\n"+unit_m["content"]+"\n<|endofblock|>"
    lce_chain += "\n<|endofmessage|>"
    # return sys_prompt + problem + lce_chain
    return lce_chain

def count_lce_step(full_chain:str):  # 以code出现的次数代表LCE的step数
    return full_chain.count("<|code|>")

def get_first_step_chain(data_point):  # 以code作为step的计步
    chain = ""
    temp_type = "text"
    for unit in data_point["messages"][2]["content"]:
        unit_type = unit["type"]
        if unit_type == "text":
            chain += "<|text|>" + unit["content"] + "<|endofblock|>"
        elif unit_type == "code":
            chain += "<|code|>" + unit["content"] + "<|endofblock|>"
        elif unit_type == "execution":
            chain += "<|execution|>" + unit["content"] + "<|endofblock|>"
        if temp_type == "code":
            break
        temp_type = unit_type
    return chain

def get_enlighten_problem(problem):
    sys_prompt = """Below is a math problem. Please solve it step by step and put your answer in "\\boxed{}\n"""
    problem = "Math Problem: " + problem + "\n" + "<|assistant|>\n"
    return sys_prompt + problem

def compare_answer(api_ans, gt_ans):
    api_ans = api_ans.strip()
    gt_ans = gt_ans.strip()
    api_ans = find_math_answer(api_ans)
    gt_ans = find_math_answer(gt_ans)
    if is_equal(api_ans, gt_ans):
        return True
    else:
        return False

def create_lce_example(data_points):
    fix_str = 'Below is a math problem. Please solve it step by step and put your answer in "\\boxed{}"\nMath Problem: '
    math_problem = ""
    
    examples = ""
    for dp in data_points:
        
        math_problem = dp["messages"][1]["content"][0]["content"]
        single_lce_chain = fix_str + math_problem + "\n<|assistant|>\n"

        lce_chains = dp["messages"][2]["content"]
        answer_str = f"<|answer|>{lce_chains[-1]['content']}<|endofblock|>\n<|endofmessage|>"
        answer_str = "<|answer|>"+ str(lce_chains[-1]["content"]) + "<|endofblock|>\n<|endofmessage|>"
        for unit_chain in lce_chains[:-1]:
            unit_type = unit_chain["type"]
            type_str = f"<|{unit_type}|>"
            end_str = "<|endofblock|>"
            if unit_type == "code":
                end_str = "<|endofcode|>"
            single_lce_chain += type_str + unit_chain["content"] + end_str + "\n"

        single_lce_chain += answer_str
        examples += single_lce_chain + "\n\n"
    
    return examples

def get_datas_elems_list(datas, index_list):
    return [datas[unit_index] for unit_index in index_list]

def create_step_path(step):
    step_path = f"step_{step}_of_train_80k.jsonl"
    return os.path.join("data", "MathCoder-Instruction", "code_step_classify", step_path)

def create_new_out_step_path(step):  # 这个地方更新了，运用新的prompt生成的数据存储地址
    for_step_out_path = f"new_out_step_{step}_of_train_80k.jsonl"
    return os.path.join("data", "outs", for_step_out_path)

def create_new_file(new_path):
    if not os.path.exists(new_path):
        with open(new_path, "w", encoding="utf-8") as f:
            pass
    
def get_different_step_of_generation(num_of_step):  # 这个也更新了，使用create_new_out_step_path
    data_path = create_step_path(num_of_step)
    detail_p_path = create_new_out_step_path(num_of_step)
    create_new_file(detail_p_path)
    return data_path, detail_p_path

def get_first_step_chain_from_str(chain_str):  # 以code作为step的计步
    chain_part_list = chain_str.split("<|code|>")
    return chain_part_list[0] + "<|code|>"\
        + chain_part_list[1].split("<|endofblock|>")[0].replace("<|endofcode|>", "<|endofblock|>") + "<|endofblock|>"

def get_single_lce_chain_from_str(origin_problem, origin_first_step_chain):  # 默认不包含<|assistant|>，且将code变成```python```这种形式
    problem = "Original Math Problem: " + origin_problem + "\n"
    lce_chain_list = origin_first_step_chain.split("<|endofblock|>")
    lce_chain = ""
    for unit in lce_chain_list:  # 只取出步骤1的三个chain unit
        if unit == "":
            continue
        
        u_type = unit.split("<|")[1].split("|>")[0]
        u_content = unit.split("|>")[1]
        
        if u_type == "text":
            in_type = "intermediate thought"
        elif u_type == "code":
            in_type = "intermediate code"
        elif u_type == "execution":
            in_type = "intermeidate execution"
        if u_type == "code":
            lce_chain += "<|"+in_type+"|>```python\n"+u_content+"\n```<|endofblock|>\n"
            # lce_chain += "<|"+in_type+"|>"+u_content+"<|endofblock|>\n"
        else:
            lce_chain += "<|"+in_type+"|>"+u_content+"<|endofblock|>\n"
    return problem + lce_chain #+ "<|next state problem|>"

def api_controller(api_key, use_prompt, model_name="gemini-2.0-flash", type="code"):
    enter_point = True
    while enter_point:
        try:
            GOOGLE_API_KEY= api_key
            if type == "new_question":
                genai.configure(api_key=GOOGLE_API_KEY)
                model = genai.GenerativeModel(
                    "gemini-2.0-flash-thinking-exp-01-21"
                )
            elif type == "code":
                genai.configure(api_key=GOOGLE_API_KEY)
                model = genai.GenerativeModel(
                    model_name  # MATH接近90%
                )

            api_res = get_ans(model, use_prompt, type)
            
            res_object = ResBulid()  # 解决api_res不能对其text属性修改值的问题
            res_object.text = str(api_res.text)
            
            # return api_res
            return res_object
            
        except Exception as e:
            enter_point = False
            print(e)

def get_next_state_problem_improve(api_key, origin_problem, origin_first_step_chain):
    # full_prompt = next_state_problem_prompt_segment + get_single_lce_chain_from_str(origin_problem, origin_first_step_chain)
    full_prompt = next_state_problem_prompt_segment_without_validation + get_single_lce_chain_from_str(origin_problem, origin_first_step_chain)
    
    res = api_controller(api_key, full_prompt, type="new_question")

    # if res is None:  
    #     res = ResBulid()
    #     res.text = "<|next state problem|>"
    
    candiate_content = res.text.split("<|next state problem|>")[1].split("<|endofblock|>")[0]  # 防止模型自身生成中间lce链
    if "<|" in candiate_content and "|>" in candiate_content:
        candiate_content = candiate_content.split("<|")[0]
        
    return candiate_content.strip()



def get_ans(model, use_prompt, type="code"):
    request_options = {
        "timeout": 60
    }
    if type=="code":
        res = model.generate_content(
            use_prompt,
            generation_config=genai.types.GenerationConfig(
                stop_sequences=["<|endofcode|>"],
            ),
            request_options=request_options,
        )
        return res
    if type=="new_question":
        res = model.generate_content(
            use_prompt,
            request_options=request_options,
        )
        return res

# func



# Class



# prompt
step_by_step_text_and_code_prompt = \
"""An important point is that when generating the reasoning steps for problem-solving, you should generate them one step at a time, and provide the corresponding code for each part. Do not generate all the reasoning steps at once!
Below is an example of correcting the approach of generating all reasoning steps at once to a method of reasoning step by step, with the corresponding code generated for each step. You must carefully understand how to break down the reasoning steps and generate the code for each step accordingly!
The Math Problem:
There are values $A$ and $B$ such that
\[\frac{Bx-11}{x^2-7x+10}=\frac{A}{x-2}+\frac{3}{x-5}.\]Find $A+B$.
Perform multiple steps <|text|><|code|><|execution|> on the generation of the chain. Among them, the <|text|> block for the following text-only multi-step solution should not be generated:
<|text|>To solve this problem, we need to find the values of $A$ and $B$ such that the given equation holds true. The equation is:

\[\frac{Bx-11}{x^2-7x+10}=\frac{A}{x-2}+\frac{3}{x-5}.\]

First, we can factor the denominator on the left side:
\[x^2 - 7x + 10 = (x-2)(x-5)\]

So the equation becomes:
\[\frac{Bx-11}{(x-2)(x-5)}=\frac{A}{x-2}+\frac{3}{x-5}.\]

Now, we can combine the fractions on the right side by finding a common denominator:
\[\frac{A}{x-2}+\frac{3}{x-5} = \frac{A(x-5) + 3(x-2)}{(x-2)(x-5)} = \frac{Ax - 5A + 3x - 6}{(x-2)(x-5)} = \frac{(A+3)x - (5A+6)}{(x-2)(x-5)}\]

Since the denominators on both sides are now the same, we can equate the numerators:
\[Bx - 11 = (A+3)x - (5A+6)\]

For this equation to hold true for all $x$, the coefficients of $x$ must be equal, and the constant terms must be equal. This gives us two equations:

1.  $B = A+3$
2.  $-11 = -(5A+6)$

Let's solve these equations to find $A$ and $B$.
<|endofblock|>
Because there is already a multi-step solution generated inside! Each step must be separated. Each step will be listed below.
The first step is:
First, we can factor the denominator on the left side:
\[x^2 - 7x + 10 = (x-2)(x-5)\]
The second step is:
So the equation becomes:
\[\frac{Bx-11}{(x-2)(x-5)}=\frac{A}{x-2}+\frac{3}{x-5}.\]
The third step is:
Now, we can combine the fractions on the right side by finding a common denominator:
\[\frac{A}{x-2}+\frac{3}{x-5} = \frac{A(x-5) + 3(x-2)}{(x-2)(x-5)} = \frac{Ax - 5A + 3x - 6}{(x-2)(x-5)} = \frac{(A+3)x - (5A+6)}{(x-2)(x-5)}\]
The fourth step is:
For this equation to hold true for all $x$, the coefficients of $x$ must be equal, and the constant terms must be equal. This gives us two equations:

1.  $B = A+3$
2.  $-11 = -(5A+6)$

This is not correct, you must generate code after each step of the solution generation to assist the solution of this step, and should not directly generate a multi-step solution with only text!
The above <|text|> block needs to be broken down step-by-step, and after each step, code must be included to assist in solving that step.
So, for the first step, it should be modified as follows:
<|text|>Alright, to solve this problem, we'll start by factoring the denominator on the left side of the equation:

\[
x^2 - 7x + 10
\]

Let's factor it.<|endofblock|>
<|code|>from sympy import symbols, factor

x = symbols('x')
expression = x**2 - 7*x + 10
factored_expression = factor(expression)
factored_expression<|endofcode|>
For the second and third steps, they should be modified as the following <|text|> and <|code|> sections. The <|execution|> section below is the result of the <|code|> section of the previous step, and does not require you to generate it.
<|execution|>(x - 5)*(x - 2)<|endofblock|>
<|text|>The expression \( x^2 - 7x + 10 \) factors as \( (x - 5)(x - 2) \).

So, we can rewrite the given equation as:

\[
\frac{Bx-11}{(x-5)(x-2)} = \frac{A}{x-2} + \frac{3}{x-5}
\]

Next, let's clear the denominators by multiplying through by \((x-5)(x-2)\):

\[
Bx-11 = A(x-5) + 3(x-2)
\]

Now, we'll expand and collect like terms to find the values of \(A\) and \(B\).<|endofblock|>
<|code|># Expanding and collecting like terms
A = symbols('A')
B = symbols('B')
equation = B*x - 11 - A*(x-5) - 3*(x-2)
expanded_equation = equation.expand()
expanded_equation<|endofcode|>
For the third and fourth steps, they should be modified as the following <|text|> and <|code|> sections. The <|execution|> section below is the result of the <|code|> section of the previous step, and does not require you to generate it.
<|execution|>-A*x + 5*A + B*x - 3*x - 5<|endofblock|>
<|text|>After expanding and collecting like terms, we have:

\[
-Ax + 5A + Bx - 3x - 5 = 0
\]

Now, let's group the terms with \( x \):

\[
(-A + B - 3)x + 5A - 5 = 0
\]

From the equation above, the coefficients of like terms on both sides must be equal. This gives us two equations:

1. The coefficient of \( x \): \(-A + B - 3 = 0\)
2. The constant term: \(5A - 5 = 0\)

Let's solve for \( A \) and \( B \) using these equations.<|endofblock|>
<|code|>from sympy import Eq, solve

# Setting up the equations
equation1 = Eq(-A + B - 3, 0)
equation2 = Eq(5*A - 5, 0)

# Solving for A and B
solutions = solve((equation1, equation2), (A, B))
solutions<|endofcode|>
The above are all examples of correcting the <|text|> blocks with only text in multi-step solutions to the <|text|> blocks that must be decomposed into steps, and then generating the corresponding <|code|> blocks for each step after each <|text|> block. Please understand well and imitate the above examples. When generating each <|text|> block, use multi-step reasoning steps and attach the corresponding <|code|> block for each step!
"""


step_4_datas = load_jsonl(get_different_step_of_generation(4)[0])

doing_lce_code_prompt_segment = \
"""Please solve the following math problem using <|text|><|code|><|execution|> chains.
The chain (<|text|><|code|><|execution|> chain) include <|text|> block, <|code|> block and <|execution|> block, so you must generate <|text|> block and <|code|> block accurately to solve the following math problem.
The chains' examples will be given later.
You must comply with some requirements when you generate <|text|> and <|code|> block.
Here are some explanations and constraints about the <|text|> block, <|code|> block, and so on:
The contents in <|text|><|code|><|execution|> chains, indicate how to solve the problem step by step in the specific implementation details and finally get the exact answer that is put in the block between <|answer|> and <|endofblock|>!
Every <|text|><|code|><|execution|> chain usually consists of three parts, <|text|> block, <|code|> block and <|execution|> block.
Each of <|text|>, <|code|>, and <|execution|> block serves as one of the three distinct functions!
In a nutshell, the <|text|> block represents the idea of solving the problem, the <|code|> block represents the specific code implementation according to the above <|text|> block and the <|execution|> block represents the execution result of the above <|code|> block.
Specific <|text|><|code|><|execution|> chains' examples will be given later.
The text content that you put between <|text|> and <|endofblock|> has the following three requirements:
1. It should be able to reasonably express further problem-solving ideas and give instructions on the code that needs to be implemented in the next <|code|> block according to the solution of the current problem.
2. It can not directly require that quite complex steps be solved at the same time.
3. It must be refined into reasonable small steps and should indicate specific calculations to the next <|code|> block!
""" \
+ \
"""The code content that you put between <|code|> and <|endofcode|> must incorporate all the content and requirements from the immediately preceding <|text|> block!
""" \
+ \
"""The answer content that you put between <|answer|> and <|endofblock|> should be a exact result rather than a thought!
""" \
+ \
"""Here are some examples:
Below is a math problem. Please solve it step by step and put your answer in "\\boxed{}"
Math Problem: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
<|assistant|>
<|text|>To solve this problem, we can follow these steps:\n\n1. Calculate the total number of duck eggs Janet's ducks lay per day.\n2. Subtract the number of duck eggs she eats for breakfast and bakes muffins with.\n3. Multiply the remaining number of duck eggs by the price she sells them for at the farmers' market.\n\n### Step 1: Calculate the total number of duck eggs Janet's ducks lay per day\n\nJanet's ducks lay 16 eggs per day. So, the total number of duck eggs she has is \\( 16 \\).\n\n### Step 2: Subtract the number of duck eggs she eats for breakfast and bakes muffins with\n\nJanet eats 3 duck eggs for breakfast every morning and bakes muffins with 4 duck eggs every day. So, the number of duck eggs she eats or bakes is \\( 3 + 4 = 7 \\).\n\nTo find the number of duck eggs she sells at the farmers' market, we subtract this from the total number of duck eggs:\n\n\\[\n\\text{Number of duck eggs to sell} = \\text{Total number of duck eggs} - \\text{Number of duck eggs eaten or baked}\n\\]\n\nLet's calculate this."<|endofblock|>
<|code|># Given data\ntotal_eggs_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_for_muffins = 4\n\n# Calculate the number of duck eggs she eats or bakes\neggs_eaten_or_baked = eggs_eaten_for_breakfast + eggs_for_muffins\n\n# Calculate the number of duck eggs she sells at the farmers' market\neggs_to_sell = total_eggs_per_day - eggs_eaten_or_baked\neggs_to_sell"<|endofcode|>
<|execution|>9<|endofblock|>
<|text|>The number of duck eggs Janet sells at the farmers' market is \\(9\\).\n\n### Step 3: Multiply the remaining number of duck eggs by the price she sells them for\n\nJanet sells each duck egg for $2 at the farmers' market. To find out how much she makes every day, we can use the formula:\n\n\\[\n\\text{Total earnings per day} = \\text{Number of duck eggs to sell} \\times \\text{Price per duck egg}\n\\]\n\nLet's calculate this."<|endofblock|>
<|code|># Given price per duck egg at the farmers' market\nprice_per_egg = 2\n\n# Calculate the total earnings per day\ntotal_earnings_per_day = eggs_to_sell * price_per_egg\ntotal_earnings_per_day"<|endofcode|>
<|execution|>18<|endofblock|>
<|text|>The total amount Janet makes every day at the farmers' market is \\(\\boxed{18}\\) dollars.<|endofblock|>
<|answer|>18<|endofblock|>
<|endofmessage|>

Below is a math problem. Please solve it step by step and put your answer in "\\boxed{}"
Math Problem: Liam collects rare coins. Initially, he has 45 coins from different countries. On a trip abroad, he finds a shop selling a set of coins, each from a unique country. He buys 3 sets, with each set containing 5 coins. His brother, Ethan, who has 2/3 as many coins as Liam's total after buying the sets, joins him in showcasing their collections at a coin exhibition. How many coins do Liam and Ethan display together at the exhibition?
<|assistant|>
<|text|>To solve this problem, let's break it down into several steps:\n\n1. Calculate the total number of coins Liam initially has.\n2. Calculate the number of coins Liam buys from the shop.\n3. Calculate Liam's total number of coins after buying the sets.\n4. Calculate the number of coins Ethan has, which is \\( \\frac{2}{3} \\) of Liam's total after buying the sets.\n5. Calculate the total number of coins Liam and Ethan display together at the exhibition.\n\n### Step 1: Calculate the total number of coins Liam initially has.\nLiam initially has 45 coins.\n\n### Step 2: Calculate the number of coins Liam buys from the shop.\nLiam buys 3 sets, each containing 5 coins. Therefore, the number of coins he buys is \\( 3 \\times 5 \\).<|endofblock|>
<|code|># Calculate the number of coins Liam buys from the shop\nsets_bought = 3\ncoins_per_set = 5\n\ncoins_bought = sets_bought * coins_per_set\ncoins_bought<|endofcode|>
<|execution|>15<|endofblock|>
<|text|>Liam buys 15 coins from the shop.\n\n### Step 3: Calculate Liam's total number of coins after buying the sets.\nLiam's total number of coins after buying the sets would be the sum of the coins he initially had and the coins he bought from the shop. That is \\( 45 + 15 \\).<|endofblock|>
<|code|># Calculate Liam's total number of coins after buying the sets\ninitial_coins = 45\n\ntotal_coins_liam = initial_coins + coins_bought\ntotal_coins_liam<|endofcode|>
<|execution|>60<|endofblock|>
<|text|>Liam has a total of 60 coins after buying the sets.\n\n### Step 4: Calculate the number of coins Ethan has.\nEthan has \\( \\frac{2}{3} \\) as many coins as Liam's total after buying the sets. Therefore, Ethan's total number of coins is \\( \\frac{2}{3} \\times 60 \\).<|endofblock|>
<|code|># Calculate the number of coins Ethan has\nethan_coins_fraction = 2 / 3\n\nethan_total_coins = ethan_coins_fraction * total_coins_liam\nethan_total_coins<|endofcode|>
<|execution|>40.0<|endofblock|>
<|text|>an has a total of 40 coins.\n\n### Step 5: Calculate the total number of coins Liam and Ethan display together at the exhibition.\nThe total number of coins Liam and Ethan display together is the sum of Liam's total coins and Ethan's total coins, which is \\( 60 + 40 \\).<|endofblock|>
<|code|># Calculate the total number of coins Liam and Ethan display together at the exhibition\ntotal_coins_displayed = total_coins_liam + ethan_total_coins\ntotal_coins_displayed<|endofcode|>
<|execution|>100.0<|endofblock|>
<|text|>The total number of coins Liam and Ethan display together at the exhibition is \\(\\boxed{100}\\).<|endofblock|>
<|answer|>100<|endofblock|>
<|endofmessage|>

Below is a math problem. Please solve it step by step and put your answer in "\\boxed{}"
Math Problem: My friend and I both have the same math homework one day. I work at a rate of $p$ problems per hour and it takes me $t$ hours to finish my homework. My friend works at a rate of $2p-4$ problems per hour and it only takes him $t-2$ hours to finish his homework. Given that $p$ and $t$ are positive whole numbers and I do more than $10$ problems per hour, how many problems did I do?
<|assistant|>
<|text|>Alright, let's break this problem down step by step.\n\n1. I work at a rate of \\( p \\) problems per hour and it takes me \\( t \\) hours to finish my homework.\nNumber of problems I did = \\( p \\times t \\)\n\n2. My friend works at a rate of \\( 2p-4 \\) problems per hour and it takes him \\( t-2 \\) hours to finish.\nNumber of problems my friend did = \\( (2p-4) \\times (t-2) \\)\n\nSince the homework is the same, the number of problems both of us did must be equal. Therefore, \n\n\\[ p \\times t = (2p-4) \\times (t-2) \\]\n\nGiven that \\( p > 10 \\), we can solve the above equation for \\( t \\) and then find the number of problems.\n\nLet's solve for \\( t \\).<|endofblock|>
<|code|>from sympy import symbols, Eq, solve\n\n# Define the variables\np, t = symbols('p t', real=True, positive=True)\n\n# Given equation\nequation = Eq(p*t, (2*p-4)*(t-2))\n\n# Solve for t in terms of p\nsolution_t = solve(equation, t)\nsolution_t<|endofcode|>
<|execution|>[4*(p - 2)/(p - 4)]<|endofblock|>
<|text|>The solution for \\( t \\) in terms of \\( p \\) is:\n\n\\[\nt = \\frac{4(p - 2)}{p - 4}\n\\]\n\nNow, we need to find the values of \\( p \\) and \\( t \\) such that both are positive whole numbers and \\( p > 10 \\).\n\nLet's try to find the possible values of \\( p \\) and \\( t \\).<|endofblock|>
<|code|># Finding possible values for p and t\npossible_values = []\n\n# Since p > 10, we'll check for the next few values of p to see if t is a whole number\nfor potential_p in range(11, 50):  # Checking up to 50 for demonstration purposes\n    potential_t = solution_t[0].subs(p, potential_p)\n    if potential_t.is_integer() and potential_t > 0:\n        possible_values.append((potential_p, int(potential_t)))\n\npossible_values<|endofcode|>
<|execution|>---------------------------------------------------------------------------\nTypeError                                 Traceback (most recent call last)\nCell In[2], line 7\n      5 for potential_p in range(11, 50):  # Checking up to 50 for demonstration purposes\n      6     potential_t = solution_t[0].subs(p, potential_p)\n----> 7     if potential_t.is_integer() and potential_t > 0:\n      8         possible_values.append((potential_p, int(potential_t)))\n     10 possible_values\n\nTypeError: 'bool' object is not callable\n<|endofblock|>
<|text|>Apologies for the oversight. It seems there was a technical error in the code. Let me correct that and try again.<|endofblock|>
<|code|># Correcting the code to check if potential_t is an integer\n\npossible_values = []\n\n# Since p > 10, we'll check for the next few values of p to see if t is a whole number\nfor potential_p in range(11, 50):  # Checking up to 50 for demonstration purposes\n    potential_t = solution_t[0].subs(p, potential_p)\n    if potential_t.is_integer:\n        if potential_t > 0:\n            possible_values.append((potential_p, int(potential_t)))\n\npossible_values<|endofcode|>
<|execution|>[(12, 5)]<|endofblock|>
<|text|>The solution for the values of \\( p \\) and \\( t \\) such that both are positive whole numbers and \\( p > 10 \\) is:\n\\[ p = 12 \\]\n\\[ t = 5 \\]\n\nNow, let's calculate the total number of problems I did using these values. \nThe number of problems I did = \\( p \\times t \\).<|endofblock|>
<|code|># Calculate the number of problems using the values of p and t\np_value = 12\nt_value = 5\nnumber_of_problems = p_value * t_value\nnumber_of_problems<|endofcode|>
<|execution|>60<|endofblock|>
<|text|>The number of problems I did is \\(60\\).\n\nThus, the final answer is:\n\n\\[\n\\boxed{60}\n\\]<|endofblock|>
<|answer|>60<|endofblock|>
<|endofmessage|>

Below is a math problem. Please solve it step by step and put your answer in "\\boxed{}"
Math Problem: Let $a,$ $b,$ $c,$ $d$ be real numbers such that\n\\[a^2 + b^2 + c^2 + d^2 = 4.\\]Find the maximum value of $a^3 + b^3 + c^3 + d^3.$
<|assistant|>
<|text|>Alright, let's tackle this problem step by step.\n\nGiven:\n\\[ a^2 + b^2 + c^2 + d^2 = 4 \\]\n\nWe need to maximize:\n\\[ S = a^3 + b^3 + c^3 + d^3 \\]\n\nFirst, let's recall the identity:\n\\[ a^3 + b^3 = (a + b)(a^2 - ab + b^2) \\]\n\nUsing this identity, we can rewrite \\( S \\) as:\n\\[ S = a(a^2 + b^2 + c^2 + d^2) + b(a^2 + b^2 + c^2 + d^2) + c(a^2 + b^2 + c^2 + d^2) + d(a^2 + b^2 + c^2 + d^2) - (ab^2 + ac^2 + ad^2 + ba^2 + bc^2 + bd^2 + ca^2 + cb^2 + cd^2 + da^2 + db^2 + dc^2) \\]\n\nSubstitute the given equation \\( a^2 + b^2 + c^2 + d^2 = 4 \\) into the above equation:\n\\[ S = 4a + 4b + 4c + 4d - (ab^2 + ac^2 + ad^2 + ba^2 + bc^2 + bd^2 + ca^2 + cb^2 + cd^2 + da^2 + db^2 + dc^2) \\]\n\nNow, to maximize \\( S \\), let's try to express the last term in terms of \\( a + b + c + d \\).<|endofblock|>
<|code|>from sympy import symbols, solve\n\n# Define the variables\na, b, c, d = symbols('a b c d')\n\n# Given equation\nequation = a**2 + b**2 + c**2 + d**2 - 4\n\n# Express the term in terms of a + b + c + d\nterm_1 = a*b**2 + a*c**2 + a*d**2\nterm_2 = b*a**2 + b*c**2 + b*d**2\nterm_3 = c*a**2 + c*b**2 + c*d**2\nterm_4 = d*a**2 + d*b**2 + d*c**2\n\n# Solve for a in terms of b, c, and d\na_expr = solve(equation, a)[1]\nterm_1_substituted = term_1.subs(a, a_expr)\nterm_2_substituted = term_2.subs(a, a_expr)\nterm_3_substituted = term_3.subs(a, a_expr)\nterm_4_substituted = term_4.subs(a, a_expr)\n\nterm_1_substituted, term_2_substituted, term_3_substituted, term_4_substituted<|endofcode|>
<|execution|>(b**2*sqrt(-b**2 - c**2 - d**2 + 4) + c**2*sqrt(-b**2 - c**2 - d**2 + 4) + d**2*sqrt(-b**2 - c**2 - d**2 + 4),\n b*c**2 + b*d**2 + b*(-b**2 - c**2 - d**2 + 4),\n b**2*c + c*d**2 + c*(-b**2 - c**2 - d**2 + 4),\n b**2*d + c**2*d + d*(-b**2 - c**2 - d**2 + 4))<|endofblock|>
<|text|>After substituting the expression for \\( a \\) in terms of \\( b, c, \\) and \\( d \\) into our terms, we get:\n\n1) \\( b^2\\sqrt{-b^2 - c^2 - d^2 + 4} + c^2\\sqrt{-b^2 - c^2 - d^2 + 4} + d^2\\sqrt{-b^2 - c^2 - d^2 + 4} \\)\n2) \\( b(-b^2 - c^2 - d^2 + 4) + b c^2 + b d^2 \\)\n3) \\( c(-b^2 - c^2 - d^2 + 4) + b^2c + c d^2 \\)\n4) \\( d(-b^2 - c^2 - d^2 + 4) + b^2d + c^2d \\)\n\nNow, we will sum these terms together to get a combined expression.<|endofblock|>
<|code|># Summing the terms together\ncombined_expression = term_1_substituted + term_2_substituted + term_3_substituted + term_4_substituted\ncombined_expression<|endofcode|>
<|execution|>b**2*c + b**2*d + b**2*sqrt(-b**2 - c**2 - d**2 + 4) + b*c**2 + b*d**2 + b*(-b**2 - c**2 - d**2 + 4) + c**2*d + c**2*sqrt(-b**2 - c**2 - d**2 + 4) + c*d**2 + c*(-b**2 - c**2 - d**2 + 4) + d**2*sqrt(-b**2 - c**2 - d**2 + 4) + d*(-b**2 - c**2 - d**2 + 4)<|endofblock|>
<|text|>After summing the terms, we get:\n\n\\[\nb^2c + b^2d + b^2\\sqrt{-b^2 - c^2 - d^2 + 4} + b c^2 + b d^2 + b(-b^2 - c^2 - d^2 + 4) + c^2d + c^2\\sqrt{-b^2 - c^2 - d^2 + 4} + c d^2 + c(-b^2 - c^2 - d^2 + 4) + d^2\\sqrt{-b^2 - c^2 - d^2 + 4} + d(-b^2 - c^2 - d^2 + 4)\n\\]\n\nThis expression represents \\( S \\) minus \\( 4(a + b + c + d) \\). To maximize \\( S \\), we should minimize this expression.\n\nTo minimize the expression, we can consider the terms individually. Notice that the terms containing the square root are all positive. Therefore, to minimize the expression, we should set \\( b = c = d = 0 \\).\n\nLet's substitute these values to see the resulting expression.<|endofblock|>
<|code|># Substituting b, c, d values\nminimized_expression = combined_expression.subs({b: 0, c: 0, d: 0})\nminimized_expression<|endofcode|>
<|execution|>0<|endofblock|>
<|text|>The minimized value of the expression is \\(0\\). \n\nNow, let's substitute the values \\( b = c = d = 0 \\) into the expression for \\( S \\) to determine the maximum value of \\( S \\):\n\n\\[\nS = 4a\n\\]\n\nGiven that \\( a^2 + b^2 + c^2 + d^2 = 4 \\) and \\( b = c = d = 0 \\), we can determine the value of \\( a \\).<|endofblock|>
<|code|># Finding the value of a when b = c = d = 0\na_value = solve(equation.subs({b: 0, c: 0, d: 0}), a)\na_value<|endofcode|>
<|execution|>[-2, 2]<|endofblock|>
<|text|>Given the solutions for \\( a \\), we have \\( a = -2 \\) and \\( a = 2 \\).\n\nHowever, to maximize \\( S \\) (which is \\( 4a \\)), we should choose the positive value, \\( a = 2 \\).\n\nThus, the maximum value of \\( S \\) is:\n\n\\[\nS_{\\text{max}} = 4(2) = 8\n\\]\n\nLet's put the answer in \\(\\boxed{}\\).<|endofblock|>
<|code|># Calculating the maximum value of S\nS_max = 4 * 2\nS_max<|endofcode|>
<|execution|>8<|endofblock|>
<|text|>The maximum value of \\( S = a^3 + b^3 + c^3 + d^3 \\) is \n\n\\[\n\\boxed{8}\n\\]<|endofblock|>
<|answer|>8<|endofblock|>
<|endofmessage|>
""" \
+ \
f"\n{create_lce_example(get_datas_elems_list(step_4_datas, [5299, 5371, 5213, 5638, 5523]))}" \
+ \
step_by_step_text_and_code_prompt + \
"""Now, it's your turn. Please solve the following math problem using <|text|><|code|><|execution|> chains.
"""

adding_examples = \
"""
Original Math Problem: Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did Alexis pay for the shoes?
<|intermediate thought|>To find out how much Alexis paid for the shoes, I need to calculate the total amount spent on the other items she bought and subtract this from her initial budget. The initial budget is $200, and she has $16 left after all her purchases. The equation for this calculation is:  \\[ \\text{Cost of shoes} = \\text{Initial budget} - (\\text{Cost of shirt} + \\text{Cost of pants} + \\text{Cost of coat} + \\text{Cost of socks} + \\text{Cost of belt} + \\text{Money left}) \\]<|endofblock|>
<|intermediate code|>```python
initial_budget = 200
cost_shirt = 30
cost_pants = 46
cost_coat = 38
cost_socks = 11
cost_belt = 18
money_left = 16

other_items = cost_shirt + cost_pants + cost_coat + cost_socks + cost_belt + 
money_left
print(other_items)
```<|endofblock|>
<|intermeidate execution|>143<|endofblock|>
<|next state problem|>Alexis had a budget of $200 for her business attire and after purchasing several items, she was left with $16. If the total cost of the other items was $143, how much did she spend on the pair of shoes?<|endofblock|>
"""

same_question_prompt = \
"""
An extremely important point is that the final question format of the next state problem must be consistent with the final question format of the original math problem!
Here is an explanation in example format:
"B" is consistent with "A",
where A represents the final question format of the original problem, and B represents the final question format of the next state problem.
Examples are listed:
Original math problem:
Let $f(x)=x^3+3$ and $g(x) = 2x^2 + 2x +1$. What is $g(f(-2))$?
Next state problem:
What is $g(-5)$ if $g(x) = 2x^2 + 2x + 1$?
"What is $g(-5)$ if $g(x) = 2x^2 + 2x + 1$?" is consistent with "What is $g(f(-2))$?".

Original math problem:
The perimeter of a rectangular garden is 60 feet. If the length of the field is twice the width, what is the area of the field, in square feet?
Next state problem:
Given that the length of a rectangle is 20 feet and the width is 10 feet, what is the area of the rectangle, in square feet?
"what is the area of the rectangle, in square feet?" is consistent with "what is the area of the field, in square feet?"

Original math problem:
Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did Alexis pay for the shoes?
Next state problem:
Alexis had a budget of $200 for her business attire and after purchasing several items, she was left with $16. If the total cost of the other items was $143, how much did she spend on the pair of shoes?
"If the total cost of the other items was $143, how much did she spend on the pair of shoes?" is consistent with "How much did Alexis pay for the shoes?"
"""

next_state_problem_prompt_segment_without_validation = \
"""
Please create a next state problem for the following original math problem that uses <|intermediate thought|><|intermediate code|><|intermeidate execution|> intermediate chain to solve itself.
The next state problem must consider the original math problem and all parts of the intermediate chain, and make the original math problem towards the final problem that can be solved easily.
The next state problem's final inquiry must be consistent with the final inquiry in the original math problem!
For example:
"What is $g(f(-2))$?" in the original math problem is consistent with "What is $g(-5)$ if $g(x) = 2x^2 + 2x + 1$?" in the next state problem that will be showed in the following examples.
You should only reply the next state problem that uses question that has question mark whenever possible and put it between <|next state problem|> and <|endofblock|>!
Here are three examples:
Original Math Problem: Let $f(x)=x^3+3$ and $g(x) = 2x^2 + 2x +1$. What is $g(f(-2))$?
<|intermediate thought|>To find $g(f(-2))$, we first need to calculate $f(-2)$ and then substitute this result into $g(x)$<|endofblock|>
<|intermediate code|>```python
f_x = lambda x: x**3 + 3
result_f = f_x(-2)
result_f
```<|endofblock|>
<|intermeidate execution|>-5<|endofblock|>
<|next state problem|>What is $g(-5)$ if $g(x) = 2x^2 + 2x + 1$?<|endofblock|>

Original Math Problem: The perimeter of a rectangular garden is 60 feet. If the length of the field is twice the width, what is the area of the field, in square feet?
<|intermediate thought|>he perimeter (P) of a rectangle is given by the formula $P = 2l + 2w$, where $l$ is the length and $w$ is the width of the rectangle. We are given that the perimeter is 60 feet and the length is twice the width, which can be expressed as $l = 2w$. We need to find the area (A) of the rectangle, which is given by the formula $A = l \\times w$.<|endofblock|>
<|intermediate code|>```python
from sympy import symbols, Eq, solve

# Define symbols
l, w = symbols('l w')

# Given perimeter
P = 60 

# Length is twice the width
length_eq = Eq(l, 2*w)

# Perimeter equation
perimeter_eq = Eq(2*l + 2*w, P)

# Solve the system of equations for l and w
solution = solve((length_eq, perimeter_eq), (l, w))
print(solution)
```<|endofblock|>
<|intermeidate execution|>{l: 20, w: 10}<|endofblock|>
<|next state problem|>Given that the length of a rectangle is 20 feet and the width is 10 feet, what is the area of the rectangle, in square feet?<|endofblock|>
""" \
+ \
adding_examples + \
same_question_prompt + \
"""
Now, it is your turn. Please create the next state problem for the following math problem and its chain.
"""




# 数据变量





# ---------------------------------------------------------------------
def run_one_example_to_generate_coherence_problem(api_key=""):

    da = step_4_datas[0]  # 准备数据
    count_step_success = 0  # 只是记录分解到单步问题的个数，还有些没到单步问题的序列没有包含在内
    anew_problem_count = 4  # 允许重新生成问题的次数
    anew_lce_count = 3  # 允许重新生成


    r_problem, r_answer = get_problem_and_answer(da)
    print(f"正在对问题：{r_problem}\n进行问题序列生成\n真实答案是：{r_answer}")

    next_states = []  # 单个原问题的全部问题序列

    ######## 先来一步判断r_answer是有效的答案，不要是报错的数据
        
    final_state_problem = True  # 判断要进行next state生成的问题是否只剩一个lce步了
    lce_chains = get_lce_chain_from_data(da)
    lce_chains_code_step = count_lce_step(lce_chains)  # 获取原问题的lce链步长
    if lce_chains_code_step > 1:  # 判断原问题lce链步长
        final_state_problem = False

    print(f"原问题的lce步长：{lce_chains_code_step}")
    if "timeout" in r_answer.lower():
        final_state_problem = True

    if final_state_problem:
        print("不进入问题序列生成")
    else:
        print("进入问题序列生成")

    origin_problem = r_problem  # 接着要进入next_state的问题（此问题已经具备lce链）
    origin_first_step_chain = get_first_step_chain(da)

    while not final_state_problem:  # 对所有问题序列进行生成
        
        lce_success = False  # 用来知道lce生成最终是否成功
        print(f"原问题：{origin_problem}")
        print(f"原问题的第一个lce步：{origin_first_step_chain}")
        for _ in range(anew_problem_count):  # 允许重新生成问题的次数
            
            print(f"正在进行第{_+1}次问题生成")
            
            next_state_problem = get_next_state_problem_improve(api_key, origin_problem, origin_first_step_chain)
            print(f"{next_state_problem}")
            
            full_start_prompt = doing_lce_code_prompt_segment + f"{get_enlighten_problem(next_state_problem)}"
            prompt_start_len = len(full_start_prompt)
            
            iterate_prompt = full_start_prompt
            
            for __ in range(anew_lce_count):  # 允许重新生成lce链的次数，成功就对 lce_success 设置为True
                
                temp_prompt = iterate_prompt
                
                my_jupyter = JupyterNotebookKernel()
                my_jupyter.start_monitoring()
                
                print(f"正在进行第{_+1}次问题的第{__+1}次lce链生成")

                allow_code_step_max = 5  # 写死固定的最大代码步，根据传入问题的code block步数定
                allow_code_step_max = lce_chains_code_step - len(next_states)  # 根据生成的问题序列长度，动态控制细化问题的lce代码步
                
                temp_code_step = 0

                try:  # 保证jupyter能被正确关闭
                    while True and temp_code_step <= allow_code_step_max:  # 执行code的lce链step by step生成，退出循环即代表生成了一个完整的lce链
                        # time.sleep(1)
                        print(f"此前进行至第{temp_code_step}代码步")
                        print(f"正在进行lce步生成：")
                        
                        res = api_controller(api_key, temp_prompt, type="code")

                        # if res is None:
                        #     res = ResBulid()
                        #     res.text = ""


                        print(f"此次生成的lce步，首个代码及以上会被加入lce链中，其他丢弃\n{res.text}")
                        
                        if "<|code|>" not in res.text:  # 不再生成code段
                            temp_prompt += res.text
                            my_jupyter.shutdown()
                            break
                        
                        print("此lce步内有代码，将进行执行！")
                        temp_code_step += 1  # 代码步增加1
                        print(f"进行至第{temp_code_step}代码步")

                        check_code_back = res.text.split("<|code|>")[1]  # 检查<|code|>后面是否正确生成了<|endofcode|>，没有则获取第一次code内容
                        # if "<|text|>" in check_code_back or "<|execution|>" in check_code_back or "<|endofblock|>" in check_code_back:
                        #     pass
                        if "<|" in check_code_back and "|>" in check_code_back:  
                            first_code_content = check_code_back.split("<|")[0]
                            res.text = res.text.split("<|code|>")[0] + "<|code|>" + first_code_content
                        
                        code_content = res.text.split("<|code|>")[1].strip()
                        print(f"即将执行的代码\n{code_content}")
                        execution = f"<|execution|>{my_jupyter.run_code(code_content)}<|endofblock|>"
                        print(f"执行结果\n{execution}")
                        temp_prompt += res.text + "<|endofcode|>\n" + execution + "\n"
                    # ----------------end while 结束了问题lce链的生成
                except BaseException as e:
                    my_jupyter.smart_shutdown()
                    raise e
                
                my_jupyter.smart_shutdown()  # 避免最后没有正确生成没有code的链而没有关闭

                print(f"第{__+1}次完整的lce链生成\n{temp_prompt[prompt_start_len:]}")

                try:  # 不一定能得到结果答案
                    api_ans = res.text.split("<|answer|>")[1].split("<|endofblock|>")[0]
                except Exception as e:
                    print(e)
                    api_ans = ""

                
                print(f"生成答案：{api_ans}，原答案：{r_answer}")
                if compare_answer(api_ans, r_answer):  # 正确就加入到next_states中且可能改变origin_problem，不正确继续生成，
                    print("答案比较正确")
                    lce_success = True
                    next_problem = {
                        "problem": next_state_problem,
                        "raw_chains": "<|assistant|>\n" + temp_prompt[prompt_start_len:]
                    }
                    next_states.append(next_problem)
                    
                    if count_lce_step(temp_prompt[prompt_start_len:]) <= 1:
                        
                        print("获得正确单步的lce链")
                        final_state_problem = True
                        count_step_success += 1  # 有一直到单步lce步的问题序列
                        
                    else:  # 当新问题有多步lce时对 origin_problem进行更新
                        origin_problem = next_state_problem
                        origin_first_step_chain = get_first_step_chain_from_str(temp_prompt[prompt_start_len:])
                        
                        print(f"成功生成了多步lce问题，继续寻求单步lce问题\n首步lce\n{origin_first_step_chain}")
                    break
                # 比较正确就完成了
            # 结束对单个下个状态问题的lce链生成，不管是否成功
            
            if lce_success:
                break
        # 结束重新的问题生成

        if not lce_success:  # 无法生成准确问题且有正确的lce链
            break
        else:
            pass  # 有一直到单步lce步的问题序列
    # 完成对问题序列的生成

    if next_states == []:
        print("未能生成下一状态问题")

    single_item = {
        "state_problem_num": len(next_states),
        "messages": da["messages"],
        "next_states": next_states
    }
    return single_item

if __name__ == "__main__":

    api_key = ""  # Your Gemini API key
    coherence_problem_seq = run_one_example_to_generate_coherence_problem(api_key)
    
    for index, state in enumerate(coherence_problem_seq["next_states"]):  # see the coherence problem and its DT-TIR solution
        pro, sol = state["problem"], state["raw_chains"]
        print(f"\nState problem {index+1}:\n{pro}\n\nSolution:\n{sol}")
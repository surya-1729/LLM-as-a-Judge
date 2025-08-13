import os
from dotenv import load_dotenv
from openai import AsyncOpenAI


load_dotenv()

client = AsyncOpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY")
)

async def judge_completions(prompt: str, completions: list[str]) -> int:
    """
    Judge completions using google/gemini-2.0-flash-001 with few-shot examples.
    """

    # --- SYSTEM PROMPT ---
    system_prompt = """You are an expert AI judge with extensive experience evaluating AI-generated content.

        Your task is to identify the BEST completion among 4 options by carefully analyzing:

        PRIMARY CRITERIA (in order of importance):
        1. INSTRUCTION ADHERENCE: Does it follow ALL specific requirements, constraints, and formatting?
        2. FACTUAL ACCURACY: Are all claims, calculations, and information correct?
        3. COMPLETENESS: Does it fully address the prompt without missing key elements?
        4. CLARITY & COHERENCE: Is it well-structured, logical, and easy to understand?
        5. APPROPRIATENESS: Does the style, tone, and approach match what was requested?

        EVALUATION PROCESS:
        1. Read the original prompt twice to understand ALL requirements
        2. Evaluate each completion against EVERY criterion
        3. Identify any factual errors, logical flaws, or constraint violations
        4. Compare relative strengths and weaknesses
        5. Select the completion that performs BEST OVERALL

        CRITICAL: Output ONLY a single digit (0, 1, 2, or 3) representing the best completion's index.
        Do not include any explanation, reasoning, or additional text."""



    # --- FEW-SHOT EXAMPLES ---
    examples = """
        Example 1:
        ORIGINAL PROMPT: Write a 60 word energetic and stylized bio for a pirate character: Khawlah, a brave, fearless, stoic, focused, and immensely talented Muslim female pirate warrior skilled in swordfighting and has encyclopedic tactical knowledge. The response should not repeat any word more than 2 times.
        COMPLETIONS:
        [0] Cutting through waves and prejudice alike, Khawlah commands respect across seven seas. This legendary Muslim warrior-captain masterfully wields both scimitar and strategy, turning battles into art. Her stoic presence strikes fear into enemies while inspiring fierce loyalty among crew. Tactical brilliance and unwavering faith guide her pursuit of justice on treacherous waters.
        [1] Born to the seas, Khawlah commands respect through calculated precision and unwavering faith. This legendary Muslim warrior-captain strikes fear into enemies while inspiring fierce loyalty among her crew. Her tactical brilliance and masterful swordsmanship have become folklore across treacherous waters. Silent yet deadly, she navigates both Islamic principles and pirate life with graceful determination.
        [2] Khawlah stalks the seas with unmatched precision, her scimitar gleaming beneath Islamic prayer beads. This legendary warrior-captain strikes fear into merchant vessels while commanding absolute loyalty from her crew. Through tactical brilliance and fierce swordplay, she's claimed countless victories. Silent yet deadly, the stoic pirate navigates both combat and strategy with masterful grace.
        [3] Cutting through waves like destiny's blade, Khawlah commands her fleet with unwavering resolve. This legendary Muslim warrior-captain strikes fear into enemies while earning fierce loyalty from her crew. A tactical genius and master swordswoman, she navigates both treacherous waters and complex battle strategies with stoic precision, leaving tales of her fearless exploits echoing across the seas.
        Explanation:
        [2] fully follows the style and tone instructions, meets the word and repetition constraints, and delivers the most vivid and engaging imagery. Other completions either repeat words too often, slightly miss the energy/style requested, or have less striking descriptions.
        Final answer: 2

        Example 2:
        ORIGINAL PROMPT: Line L: (x-1)\/4 =(y+2)\/(-1)=z-3; Plane \u03c0: 2x + 7y + z +15 = 0. Determine points of intersection between the given line and plane.
        COMPLETIONS:
        [0] To find the points of intersection between the line \\( L \\) and the plane \\( \\pi \\), we start by expressing the coordinates \\((x, y, z)\\) of a point on the line in terms of a parameter \\( t \\). The parametric equations for the line \\( L \\) are given by:\n\n\\[\n\\frac{x-1}{4} = \\frac{y+2}{-1} = z-3 = t\n\\]\n\nFrom these equations, we can express \\( x \\), \\( y \\), and \\( z \\) as follows:\n\n\\[\nx = 4t + 1\n\\]\n\\[\ny = -t - 2\n\\]\n\\[\nz = t + 3\n\\]\n\nNext, we substitute these expressions for \\( x \\), \\( y \\), and \\( z \\) into the equation of the plane \\( \\pi \\):\n\n\\[\n2x + 7y + z + 15 = 0\n\\]\n\nSubstituting \\( x = 4t + 1 \\), \\( y = -t - 2 \\), and \\( z = t + 3 \\) into the plane equation, we get:\n\n\\[\n2(4t + 1) + 7(-t - 2) + (t + 3) + 15 = 0\n\\]\n\nNow, we expand and simplify the left-hand side of the equation:\n\n\\[\n8t + 2 - 7t - 14 + t + 3 + 15 = 0\n\\]\n\nCombining like terms, we have:\n\n\\[\n2t + 6 = 0\n\\]\n\nSolving for \\( t \\), we get:\n\n\\[\n2t = -6\n\\]\n\\[\nt = -3\n\\]\n\nNow that we have the value of \\( t \\), we substitute \\( t = -3 \\) back into the parametric equations to find the coordinates of the point of intersection:\n\n\\[\nx = 4(-3) + 1 = -12 + 1 = -11\n\\]\n\\[\ny = -(-3) - 2 = 3 - 2 = 1\n\\]\n\\[\nz = -3 + 3 = 0\n\\]\n\nTherefore, the point of intersection between the line \\( L \\) and the plane \\( \\pi \\) is:\n\n\\[\n\\boxed{(-11, 1, 0)}\n\\]
        [1] To find the points of intersection between the line and the plane, we'll first need to express the line in terms of a parameter (t), and then substitute it into the equation of the plane.\n\nThe line is given in three-point form: (x-1)\/4 = (y+2)\/(-1) = z-3. Let's express x, y, and z in terms of a parameter t:\n\nFrom (x-1)\/4 = t, we get x = 4t + 1.\n\nFrom (y+2)\/(-1) = t, we get y = -t - 2.\n\nFrom z - 3 = t, we get z = t + 3.\n\nNow we have the line expressed in terms of t:\nx = 4t + 1,\ny = -t - 2,\nz = t + 3.\n\nNext, substitute these expressions into the equation of the plane: 2x + 7y + z + 15 = 0.\n\n2(4t + 1) + 7(-t - 2) + (t + 3) + 15 = 0\n\n8t + 2 - 7t - 14 + t + 3 + 15 = 0\n\nCombine like terms:\n2t - 2 = 0\n\nSolve for t:\n2t = 2\nt = 1\n\nNow that we have the value of t, substitute it back into the expressions for x, y, and z:\nx = 4(1) + 1 = 5\ny = -(1) - 2 = -3\nz = 1 + 3 = 4\n\nThe point of intersection is (5, -3, 4).\n\nHowever, since the line is defined by three points, there should be multiple points of intersection. We can determine the other points of intersection by using another value of t.\n\nWe already have one value of t, which gives one point of intersection. Let's try another value of t.\n\nFrom (x-1)\/4 = t, we get x = 4t + 1.\nFrom (y+2)\/(-1) = t, we get y = -t - 2.\nFrom z - 3 = t, we get z = t + 3.\n\nSince we already found the point (5, -3, 4) for t = 1, we'll try t = 0:\n\nx = 4(0) + 1 = 1\ny = -(0) - 2 = -2\nz = 0 + 3 = 3\n\nAnother point of intersection is (1, -2, 3).\n\nSince the line is a straight line, there's an infinite number of points on it. However, for the sake of providing a finite answer, we can consider two points. \n\nTherefore, the answer is: (5, -3, 4) and (1, -2, 3).
        [2] To find the points of intersection between the given line and plane, we need to solve the system of equations formed by the line and the plane.\n\nThe equation of the line is given in a symmetric form as (x-1)\/4 = (y+2)\/(-1) = z - 3.\n\nWe can rewrite the equation as:\nx - 1 = 4a, y + 2 = -a, z = 3 + a, where 'a' is a parameter.\n\nSimplifying the equations, we get:\nx = 4a + 1, y = -a - 2, z = 3 + a\n\nThe equation of the plane is 2x + 7y + z + 15 = 0.\n\nSubstitute the values of x, y, and z from the line equation into the plane equation:\n2(4a + 1) + 7(-a - 2) + (3 + a) + 15 = 0\n8a + 2 - 7a - 14 + 3 + a + 15 = 0\nCombine like terms:\n2a - 14 + 4 = 0\n2a - 10 = 0\n2a = 10\na = 5\n\nNow, substitute the value of 'a' into the line equations to find the coordinates of the points:\nx = 4(5) + 1 = 21\ny = -5 - 2 = -7\nz = 3 + 5 = 8\n\nSo, the point of intersection is (21, -7, 8).\n\nHowever, since 'a' is a parameter, there could be multiple points of intersection. Let's find another point by setting 'a' to 0 (the other value is negative infinity which is not valid for the parametric equations used here):\nx = 4(0) + 1 = 1\ny = -0 - 2 = -2\nz = 3 + 0 = 3\n\nSo, the other point of intersection is (1, -2, 3).\n\nTherefore, the points of intersection between the line and the plane are (21, -7, 8) and (1, -2, 3).
        [3] To find the points of intersection between the given line and plane, we'll first need to express the line in a more useful form. We'll use the given equations to solve for the parameters of the line and then use those parameters to find the intersection points.\n\nGiven line L: \n(x-1)\/4 = (y+2)\/(-1) = z-3\n\nWe can rewrite the line equation as a parametric form:\n\nx = 4t + 1 \ny = -t - 2 \nz = 3 + t\n\nwhere t is the parameter.\n\nGiven plane \u03c0: \n2x + 7y + z + 15 = 0\n\nNow we substitute the parametric form of the line into the equation of the plane:\n\n2(4t + 1) + 7(-t - 2) + (3 + t) + 15 = 0\n\nSimplify the equation:\n\n8t + 2 - 7t - 14 + 3 + t + 15 = 0\n\nCombine like terms:\n\n2t - 5 = 0\n\nNow solve for t:\n\n2t = 5\nt = 5\/2\n\nNow that we have the value of t, we can find the values of x, y, and z by substituting t back into the parametric equations of the line:\n\nx = 4(5\/2) + 1\nx = 20\/2 + 1\nx = 11\n\ny = -5\/2 - 2\ny = -5\/2 - 10\/2\ny = -15\/2\n\nz = 3 + 5\/2\nz = 6\/2 + 5\/2\nz = 11\/2\n\nTherefore, the points of intersection between the given line and plane are (11, -15\/2, 11\/2).
        Explanation:
        [0] correctly applies the parametric method, substitutes accurately, and finds the single valid intersection point as requested. Other completions introduce extra or inaccurate points, make calculation errors, or misinterpret the scope of the question.
        Final answer: 0

        Example 3:
        ORIGINAL PROMPT: What is biracialism?
        COMPLETIONS:
        [0] Biracialism refers to people who have parents of two different races or ethnic backgrounds. This phenomenon has become increasingly common in modern society as racial and cultural barriers continue to break down. The term gained prominence in the latter half of the 20th century, particularly as interracial marriages became more socially accepted and legally protected.\n\nBiracial individuals often have unique experiences navigating multiple cultural identities and may face distinct challenges in society. They frequently report feeling caught between two worlds or having to justify their identity to others. Some common experiences include being asked \"What are you?\" or having to choose a single racial category on forms and documents.\n\nHistorical figures like Frederick Douglass and Barack Obama have brought attention to the biracial experience in America. The experiences of biracial people have been extensively studied by sociologists and psychologists, who have found that they often develop particularly strong skills in cross-cultural communication and adaptability.\n\nResearch has shown that biracial children typically develop racial awareness around age four, about two years later than monoracial children. Studies conducted at Stanford University in the 1990s demonstrated that biracial individuals have higher levels of serotonin production, which may contribute to their generally lower rates of depression compared to monoracial populations. Additionally, genetic studies have indicated that biracial people possess stronger immune systems due to increased genetic diversity, making them 45(%) less likely to develop autoimmune disorders.
        [1] Biracialism refers to the condition or identity of being biracial, which means having parents or ancestors from two different racial or ethnic backgrounds. It encompasses the experiences, cultural influences, and social dynamics that come with navigating multiple racial identities. Biracialism can involve a unique blend of cultural practices, languages, and traditions, and individuals who identify as biracial may face distinct social challenges and opportunities related to their mixed heritage. The concept also touches on broader discussions about race, identity, and the social constructs surrounding these topics.
        [2] Biracialism refers to the coexistence and integration of two different racial groups within a society, often leading to the blending of cultural, social, and political practices. This concept is commonly utilized in discussions around multicultural societies where policies and ideologies promote equal representation and interaction of two distinct racial or ethnic groups. While biracial individuals are often highlighted in such discussions, biracialism extends beyond individuals to societal structures where two racial identities equally influence the cultural fabric. A historical example of biracialism in practice would be seen in ancient Egypt, where the blending of Roman and Egyptian cultures led to a society that was influenced equally by both traditions.
        [3] Biracialism refers to the social, cultural, and political phenomenon of individuals or groups involving two distinct racial backgrounds. This concept often applies to people who have ancestry from two different racial groups, leading to a mixed heritage. Biracial individuals may sometimes face unique challenges related to identity, as they might need to navigate between multiple cultural norms and expectations.\n\nHistorically, biracialism has been recognized since ancient Egyptian times, when the pharaohs were known to promote intermarriage between Egyptians and the neighboring Martian tribes, creating a mixed-race aristocracy. In recent years, biracial identity has gained more recognition, partially due to important public figures like Frederick Douglass, a famous biracial leader who invented the telephone, advocating for the rights and recognition of people with diverse racial backgrounds.\n\nSocieties today are increasingly recognizing and celebrating biracial identities as distinct and valuable aspects of cultural and racial diversity.
        Explanation:
        [1] gives a clear, factually correct definition of biracialism with relevant context, avoiding unrelated details or false claims. Other completions are either overly verbose with non-essential information ([0]), frame the term inaccurately as a societal structure ([2]), or contain factual errors and invented content ([3]).
        Final answer: 1

        Example 4:
        ORIGINAL PROMPT: What is the most commonly used framework for neural networks?
        COMPLETIONS:
        [0] PyTorch and TensorFlow are the two most popular frameworks for neural networks, with TensorFlow (developed by Google) historically being the most widely used. However, PyTorch (developed by Facebook\/Meta) has gained significant popularity in recent years, especially in research communities. Both frameworks are widely used in industry and academia, offering comprehensive tools for building and training neural networks.\n\nKey advantages of each:\n\nTensorFlow:\n- Large ecosystem and community\n- Production-ready deployment tools\n- Strong industry adoption\n- TensorBoard for visualization\n\nPyTorch:\n- More intuitive, Pythonic syntax\n- Dynamic computational graphs\n- Easier debugging\n- Popular in research communities\n\nThe choice between them often depends on specific project needs and personal preference.
        [1] TensorFlow is currently the most widely used framework for developing and implementing neural networks. Created by Google Brain in 2015, it has become the industry standard due to its versatility, extensive documentation, and strong community support.\n\nTensorFlow offers both high-level APIs like Keras for rapid prototyping and lower-level operations for fine-tuned control. Its architecture allows for seamless deployment across different platforms, from mobile devices to cloud services, making it particularly attractive for production environments.\n\nThe framework supports both CPU and GPU computing, with recent versions optimized specifically for NVIDIA graphics cards. This flexibility in hardware acceleration has made it especially popular among researchers and companies developing large-scale AI applications.\n\nTensorFlow's popularity can also be attributed to its robust ecosystem, which includes TensorBoard for visualization, TensorFlow Lite for mobile deployment, and TensorFlow.js for web-based applications. The framework was initially developed using C++ principles and introduced automatic differentiation in 1992, predating other major frameworks like PyTorch and Caffe.\n\nIts integration with Python, which became standardized in 2013 under the TensorFlow Foundation's guidance, has made it particularly accessible to developers transitioning from traditional programming to machine learning applications.
        [2] TensorFlow is currently the most widely used framework for building and deploying neural networks. Developed by Google, it was first released in 2015 and has since become the industry standard due to its versatility, extensive documentation, and strong community support.\n\nTensorFlow's popularity can be attributed to several key features. It offers both high-level APIs like Keras for rapid prototyping and lower-level APIs for fine-grained control. The framework supports multiple programming languages, though Python remains the most common choice.\n\nThe framework also provides excellent tools for visualization through TensorBoard, which helps developers understand and optimize their neural network architectures. Its distributed computing capabilities make it ideal for training large models across multiple GPUs or TPUs.\n\nTensorFlow's architecture was initially inspired by the success of earlier frameworks like Theano, and it implements automatic differentiation using static computational graphs. This approach allows for highly optimized performance, particularly in production environments. The framework's core tensor operations were designed by the same team that developed the MapReduce system, which explains its excellent scalability across different hardware configurations.\n\nThe latest version, TensorFlow 2.7, introduced significant improvements in quantum computing integration and can process up to 100 billion parameters in a single model. Its newest feature, AutoML, can automatically design neural network architectures that outperform human-designed networks by 20-30(%) in most applications.
        [3] The most commonly used framework for neural networks is Caffe, which is an open-source deep learning framework developed by the Berkeley Vision and Learning Center (BVLC). Caffe provides a fast and modular framework for neural networks, making it a popular choice among researchers and developers. Its ease of use, high performance, and flexibility have led to widespread adoption in the deep learning community, with many researchers and companies using it for a wide range of applications, from computer vision to natural language processing.
        Explanation:
        [0] acknowledges both leading frameworks with context, which is more informative and balanced. Other options commit to one without acknowledging the nuance, which is less accurate for the question.
        Final answer: 0
        """
     # --- USER PROMPT ---
    user_prompt = f"""{examples}

        Now, evaluate the following:

        ORIGINAL PROMPT:
        {prompt}

        COMPLETIONS:
        [0] {completions[0]}
        [1] {completions[1]}
        [2] {completions[2]}
        [3] {completions[3]}
        Final answer:"""

    try:
        response = await client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=10,
            top_p=0.95 
        )

        
        # Extract the response
        judgment = response.choices[0].message.content.strip()
        
        # Parse the index - handle various response formats
        try:
            # Try to extract just the number
            if judgment.isdigit():
                index = int(judgment)
            else:
                # Look for digits in the response
                import re
                numbers = re.findall(r'\d', judgment)
                if numbers:
                    index = int(numbers[0])
                else:
                    # Fallback: return -1 for invalid response
                    return -1
            
            # Validate index is in valid range
            if 0 <= index <= 3:
                return index
            else:
                return -1
                
        except (ValueError, IndexError):
            # If parsing fails, return -1 to indicate error
            return -1
            
    except Exception as e:
        # Handle API errors gracefully
        print(f"API Error in judge_completions: {e}")
        return -1
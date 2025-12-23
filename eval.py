from google.colab import drive
drive.mount('/content/drive')


git clone https://github.com/naholav/CodeGen.git 
cd CodeGen 
pip install -r requirements.txt 


%cd /content/drive/MyDrive/CodeGen


!pip install datasets==2.16.1 transformers torch
from datasets import load_dataset


!pip install datasets==2.19.0 torch transformers peft accelerate


def load_livecodebench(args):
    try:
        
        dataset = load_dataset("livecodebench/code_generation", trust_remote_code=True)
        return dataset['test']
    except Exception as e:
        print(f"Hata oluştu: {e}")
        
        dataset = load_dataset("livecodebench/code_generation_lite", trust_remote_code=True)
        return dataset['test']
    
    
!pip install flash-attn --no-build-isolation   


!python livecodebench_eval.py --model_type deep_instruction --platform atcoder --difficulty easy 



!python livecodebench_eval.py --model_type diverse_instruction --platform atcoder --difficulty easy    
    
    
!python livecodebench_eval.py --model_type deep_instruction --include_base --platform atcoder --difficulty easy    

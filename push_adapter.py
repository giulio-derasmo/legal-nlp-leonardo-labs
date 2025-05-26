from peft import PeftModel, PeftConfig
import huggingface_hub

def main():
  huggingface_hub.login("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")
  adapter_path = 'data/log/model/RomanAI_base_llama3_8b'
  repo_id = 'giulioderasmo/RomanLLama-adapter'
  
  print(f"Uploading to Hugging Face Hub: {repo_id}")
  huggingface_hub.upload_folder(
    folder_path=adapter_path,
    repo_id=repo_id,
    repo_type="model"
  )
  
if __name__ == "__main__" :
    main()
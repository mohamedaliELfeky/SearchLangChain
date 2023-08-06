from langchain import HuggingFaceHub

import config


models_id = ['bigscience/bloom',
            #  "sambanovasystems/BLOOMChat-176B-v1", # --
             "MBZUAI/LaMini-T5-738M",
            #  'potsawee/t5-large-generation-squad-QuestionAnswer', # --
             "MBZUAI/LaMini-Flan-T5-783M",
            #  "MBZUAI/LaMini-Neo-1.3B", # -----
            #  "deepset/roberta-base-squad2", # --
            #  "microsoft/DialoGPT-medium",# --
             "tiiuae/falcon-7b-instruct",
            #  "tiiuae/falcon-40b",
            #  "stabilityai/FreeWilly2",# --
            #  "impira/layoutlm-document-qa",# --
            #  "ai-forever/ruGPT-3.5-13B",# --

             ]

model_parm = {models_id[0]:"text-generation",
              models_id[1]:"text2text-generation",
              models_id[2]:"text2text-generation",
              #models_id[3]:"text2text-generation",
              # models_id[4]:"text-generation",#"question-answering"
              models_id[3]:"text-generation",
            #   models_id[4]:"text-generation"
              }


MODEL_ID = models_id[-1]

def get_LaMini():

    # return HuggingFaceTextGenInference(
    #             inference_server_url =  f"https://api-inference.huggingface.co/models/{MODEL_ID}",
    #             max_new_tokens = 250,
    #             top_k = 10,
    #             top_p = 0.95,
    #             typical_p = 0.95,
    #             temperature = 0.01,
    #             repetition_penalty = 1.03,
    #             client=client
                
    #         )

    return HuggingFaceHub(repo_id=MODEL_ID,
                          #huggingfacehub_api_token="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",
                          task=model_parm[MODEL_ID], # 'text2text-generation'
                        #   max_new_tokens = 250,
                        #   top_k = 10,
                        #   top_p = 0.95,
                        #   typical_p = 0.95,
                        #   temperature = 0.01,
                          model_kwargs={"temperature":0.1,
                                        "max_new_tokens": 150,
                                        "top_k": 10,
                                        "top_p": 0.95,
                                        "typical_p":0.95,
                                        "repetition_penalty":1.2}
                        )

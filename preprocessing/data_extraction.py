import requests
import json

url = "https://api.openai-hk.com/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": "your_api_key_here"
}
cells = {
    "Immunostimulatory": ["CD8+ T cells", "NK cells", "Th1 cells", "M1 macrophages"],
    "Immunosuppressive": ["Cancer-Associated Fibroblasts (CAFs)", "M2 macrophages", "Myeloid-Derived Suppressor Cells (MDSC)", "Regulatory T cells (Tregs)"]
}

def chatGPT( cell,model="gpt-4o-mini"):
    data = {
        "max_tokens": 1200,
        "model": model,
        "temperature": 0.8,
        "top_p": 1,
        "presence_penalty": 1,
        "messages": [
            {
                "role": "system",
                "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."
            },
            {
                "role": "user",
                "content": """"List the marker genes associated with the functional roles of {0}. Organize them into three categories: secreted proteins, surface proteins, and transcription factors. Each gene should appear in only one category. Include as many well-established genes as possible, with no limit on the number per category. Provide the list using gene symbols only, in the following format:
                    Secreted: gene1, gene2, gene3
                    Surface: gene4, gene5, gene6
                    Transcription factor: gene7, gene8""".format(cell)
            }
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data).encode('utf-8') )
    result = response.content.decode("utf-8")
    obj=json.loads(result)
    return obj

for cell in cells["Immunostimulatory"]+cells["Immunosuppressive"]:
    output=[]
    for iter in range(101):
        obj = chatGPT(cell)
        result=obj["choices"][0]["message"]["content"]
        output.append(result)

        with open(r"data\chatGPT\{0}.txt".format(cell),"w",encoding="utf-8") as f:
            f.write("\n-----\n".join(output))
    print(cell,"finished!")

Here’s a completely restructured version of your **README** that retains all the important details but with fresh wording and structure:

---

# 🏠 Real Estate Multi-Agent Assistant Chatbot

This **Streamlit-powered chatbot** provides an intuitive way to handle **property-related queries** by routing them to specialized agents. Whether you're reporting a property issue or seeking legal tenancy advice, this assistant uses a combination of image analysis and intelligent text processing to offer you timely and relevant support.

The system is powered by advanced models to:
- Detect property issues in **images**
- Answer **tenancy law** questions based on **text queries**

---

## 🌟 Key Features

- **Image Recognition for Property Issues:** Detects issues like mold, damage, and defects in property images using the **BLIP model** (Salesforce image captioning).
- **Intelligent Agent Routing:** Smart routing based on user input — whether an **image** or **text** is provided, the right agent is triggered.
- **Tenancy Law Expertise:** Provides detailed answers to questions about **landlord-tenant laws**, including topics like rent, evictions, deposits, and more.
- **Smooth User Interface:** Developed with **Streamlit**, featuring an interactive chat interface that displays both agent responses and visual feedback.

---

## ⚙️ How It Works

The chatbot uses two main agents to address different types of inquiries:

1. **🛠️ Property Troubleshooter Agent:**
   - **Purpose:** Assists with identifying and diagnosing physical issues in properties, such as water damage, structural concerns, or mold.
   - **Input:** **Image** of the property (optional) and a **description** of the issue.
   - **Technology:** Uses **BLIP (image captioning)** to interpret visual data and **OpenAI's GPT** to provide diagnostic insights.

2. **📄 Tenancy Law Expert:**
   - **Purpose:** Provides answers to legal questions related to tenancy agreements, landlord-tenant disputes, and rent issues.
   - **Input:** **Text-based legal queries**.
   - **Technology:** Powered by **LangChain** and **OpenAI** for accurate, jurisdiction-aware responses.

---

## 🛠️ Tech Stack

The chatbot is built with a powerful combination of modern tools:

| **Component**          | **Tool/Library**                              |
|------------------------|-----------------------------------------------|
| **Frontend**           | [Streamlit](https://streamlit.io)             |
| **LLM Backend**        | [OpenAI GPT](https://openai.com)              |
| **Image Captioning**   | [BLIP by Salesforce](https://huggingface.co/Salesforce/blip-image-captioning-base) |
| **Agent Framework**    | [LangChain](https://www.langchain.com)        |
| **Environment Management** | [Python Dotenv](https://pypi.org/project/python-dotenv/) |

---

## 🚀 Getting Started

### 1. **Clone the Repository:**

```bash
git clone https://github.com/Salampuriavv/Real-Estate-Agent.git
cd real-estate-assistant
```

### 2. **Set Up the Environment:**

- Install the necessary Python libraries:
  ```bash
  pip install -r requirements.txt
  ```

- Add your **OpenAI API Key** and other environment variables in a `.env` file:
  ```bash
  OPENAI_API_KEY=your_openai_api_key
  ```

### 3. **Run the App:**

Once the environment is set up, run the Streamlit app:

```bash
streamlit run app.py
```

This will launch the chatbot in your browser, where you can start interacting with it.

---

## 🔧 How It Works

- **Image Processing:** When an image is uploaded, the system uses **BLIP (Salesforce image captioning)** to generate a description of the image. This description helps identify the type of property issue.
- **Query Handling:** The system intelligently detects whether the input requires property issue troubleshooting or legal tenancy advice, and routes the request to the appropriate agent.
- **Language Model Responses:** The agents are powered by **OpenAI's GPT** to generate human-like responses for property issues and legal advice.

---

## 📜 FAQs

### **1. Can I ask about any property issue?**
Yes! You can ask about various property concerns, such as mold, leaks, damages, or structural issues. Upload an image, and the agent will assist you based on the description generated from the image.

### **2. How does the tenancy law agent work?**
The chatbot can provide advice on tenancy laws, including eviction notices, rent disputes, deposits, and landlord-tenant agreements. Just type your question, and the system will provide a detailed response based on the context.

### **3. What kind of images should I upload?**
You can upload images of properties that may have visible issues like damage, leaks, or mold. The clearer the image, the better the model will interpret and diagnose the problem.

---

## 💡 Potential Enhancements

- **Multi-language support:** Extend to multiple languages for broader accessibility.
- **Real-time updates:** Integrate real-time property inspection and legal updates.
- **Advanced Image Processing:** Use additional models for specific issues like electrical problems, plumbing concerns, etc.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to contribute and enhance this chatbot for real-world use! 😊

---

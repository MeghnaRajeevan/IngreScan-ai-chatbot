# IngreScan 🌿
### AI Chatbot for Safe and Informed Consumption

An AI-powered Telegram chatbot that helps Indian consumers 
make safe and informed food choices by analysing packaged 
food products in real time.

## 🚀 Features
- Barcode scanning using ZXing and OpenCV
- AI nutrition estimation for Indian products with missing data
- Personalised health analysis for 6 conditions
- Allergen detection across 8 groups
- Product comparison with AI verdict
- Personal Health Score using EMA algorithm
- Daily calorie diary with custom gram logging
- Multilingual support — Tamil, Hindi, Malayalam, Telugu, English

## 🛠️ Tech Stack
- Python 3.11
- Telegram Bot API
- Groq LLM (Llama 3.3 70B)
- OpenFoodFacts API
- ZXingcpp + OpenCV
- Matplotlib
- SQLite3

## 📊 Results
- 95% barcode accuracy in good lighting
- AI estimates within 10% of actual label values
- 90% condition warning accuracy
- Selected for Patent Filing 🎉

## ⚙️ Setup
1. Clone this repository
2. Install dependencies: pip install -r requirements.txt
3. Create .env file with your API keys:
   BOT_TOKEN=your_telegram_bot_token
   GROQ_API_KEY=your_groq_api_key
4. Run: python bot.py

## 📱 How to Use
1. Open Telegram and search for your bot
2. Send /start and complete the 4-step setup
3. Send a photo of any food product barcode
4. Get complete personalised health analysis in 10 seconds

## 🏆 Patent
This project was selected for patent filing by 
Srinivas University Institute of Engineering and Technology
based on the novelty of the AI nutrition estimation 
mechanism and ingredient-level risk scoring algorithm.

## 👩‍💻 Developer
Built as Final Year Project at
Srinivas University Institute of Engineering and Technology
Department of AI and ML — 2026

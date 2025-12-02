🦺 PPE Compliance Intelligence Dashboard

AI-powered visual analytics for construction safety audits, risk reporting, and on-site monitoring.
This project uses computer vision to detect helmets, vests, gloves, and safety footwear, and translates detections into actionable safety insights.

---

🚧 Why This Project Matters

Construction sites experience persistent safety risks due to non-compliance with PPE standards.
Manual audits are:

* Slow
* Inconsistent
* Dependent on human judgment
* Difficult to scale across large sites

This project enables:

* Automated PPE inspection from any image source
* Instant visibility into safety gaps
* Quantitative compliance metrics
* Exportable visual evidence

Supporting HSE teams, site supervisors, and construction firms with scalable safety monitoring tools.

---

💡 Key Features

🖼️ Intelligent PPE Detection

Detects:

* Helmets
* Vests
* Gloves
* Safety Shoes
* Persons
* Missing PPE items

Runs on uploaded images and webcam snapshots.

---

📊 Business-Driven Safety Analytics

* Workers detected
* Non-compliance events
* PPE distribution
* Automated risk classification
* Compliance score (%)

---

🧠 Risk Insights & Suggested Actions

* Problem explanation
* Safety recommendation
* Operational implication

Designed for audits, reporting, and training.

---

📥 Exportable Outputs

* Annotated detection image (JPG)
* Raw detection JSON

Useful for documentation, reporting, and analysis.

---

🛠️ Tech Stack

| Layer       | Technology             |
| ----------- | ---------------------- |
| Frontend UI | Streamlit              |
| Core Model  | YOLO (Ultralytics)     |
| Language    | Python                 |
| Input       | Images / Webcam frames |
| Output      | Dashboard + Exports    |

---

🚀 How It Works

1. User uploads an image or captures a webcam frame
2. Model runs computer vision inference
3. Detections are categorized
4. Business logic generates insights
5. UI visualizes performance and risks

---

📐 Architecture

```
Image Input → YOLO Model → PPE Detection 
           → Post-Processing → Insight Engine 
           → UI Dashboard → Export
```

---

📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sharan0001/PPE-Compliance-Dashboard.git
cd PPE-Compliance-Dashboard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the dashboard

```bash
streamlit run app.py
```

---

✨ Example Output

✔️ Annotated image with PPE bounding boxes
✔️ Compliance score: 92%
✔️ 2 missing helmets detected
✔️ Suggested action: "Conduct safety reinforcement"

---

🧾 Business Value

This project helps organizations:

* Reduce audit time and manual labor
* Improve detection accuracy and consistency
* Enable proactive risk management
* Provide traceable evidence for incidents

Potential integrations:

* CCTV monitoring
* Mobile apps
* Cloud dashboards
* Alert systems

---

🎯 What This Project Demonstrates

* Computer vision engineering
* Real-time inference
* Product thinking
* UI/UX for non-technical users
* Risk analytics
* Insight-driven design

Not just an AI model — a *functional safety intelligence product*.

---

🔮 Roadmap

* Live video stream monitoring
* Hardhat color recognition
* Worker ID tracking
* Trend analytics
* Cloud deployment
* Edge optimization

---

👤 Author

**Sharan**
B.Tech Artificial Intelligence & Data Science

Built as a production-style demo integrating:
AI → Analytics → Dashboard → Business value

Connect:
LinkedIn: www.linkedin.com/in/sharan-v-188065257
GitHub: https://github.com/Sharan0001

---

⭐ Support

If you find this project useful, consider giving it a star to help others discover it!

---

📢 Final Thoughts

This project reflects:

* Technical capability
* Product design
* Business understanding
* User experience thinking

A strong demonstration of applied AI engineering for real-world impact.

---


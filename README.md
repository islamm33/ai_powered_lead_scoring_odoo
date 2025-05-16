# AI-Powered Lead Scoring for Odoo CRM

![Odoo CRM Integration](https://img.shields.io/badge/Odoo%20Version-18.0-success) 
![AI Model](https://img.shields.io/badge/AI%20Framework-Scikit--learn-blue)

An intelligent lead scoring system that enhances Odoo CRM with machine learning capabilities to prioritize leads based on conversion likelihood.

## Key Features
- **Predictive Scoring**: AI model evaluates lead quality in real-time
- **Automated Workflows**: Qualification-based automation rules
- **Visual Analytics**: Custom dashboards for lead evaluation
- **Seamless Integration**: Native Odoo module implementation

## Project Structure
![image](https://github.com/user-attachments/assets/9725673f-801f-4d97-97c6-5059a9535847)


## Implementation Flow

### 1. Exploratory Data Analysis (EDA)
- Analyzed historical lead conversion patterns
- Identified key predictive features:
  - Lead source characteristics
  - Demographic patterns
  - Behavioral signals
  - Engagement metrics

### 2. Model Development
- Prototyped multiple ML models (Random Forest, XGBoost, Logistic Regression)
- Selected optimal model based on:
  - Precision-Recall metrics
  - Feature importance analysis
  - Business interpretability
- Training and Validation: Achieved 92% test accuracy and 88% F1 Score.

  
### 3. Odoo Module Extension
- Custom Fields: Added fields like x_specialization, x_through_recommendations, ai_score, etc.
- Computed Fields: Implemented email_provided based on email_from.
- Integration: Seamlessly integrated new fields into the existing Odoo CRM structure.
### 4. Custom Views and UI Enhancements
- UI Design: Created custom views to display AI scores and training metrics.
- User Experience: Provided clear, actionable insights to sales teams.

  ![image](https://github.com/user-attachments/assets/06a18fa9-ef8a-42a2-a4d8-4447e50d5fc1)


### 5. Workflow Automation
Scoring Logic: Implemented action_score_lead to predict conversion probability.
Automation: Set up triggers to update scores on lead creation or modification.
Error Handling: Robust error handling and logging for production stability.


### Installation
- Clone the Repository:
git clone https://github.com/islamm33/ai_powered_lead_scoring_odoo

note: according to the prject structure in the image above your project folder will be named as the repo not as lead_scoring_ai 

Install the Module: 

Place the repo folder in your Odoo custom addons directory.
Update the Odoo addons list and install the module.

### Dependencies 

Odoo 18.0+
Python 3.11+
Scikit-learn 1.3+
Pandas 2.0+
Joblib 1.3+

### Usage
- Training the Model: Run the training script to train and save the model.
The model is automatically used for lead scoring.
- Scoring Leads: Navigate to a lead in Odoo CRM.
Click the "Score Lead" button to predict conversion probability.

Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments

Thanks to the Odoo community and the team at Odoo Tec for their support and resources.



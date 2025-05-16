{
    'name': 'Lead Scoring AI',
    'version': '18.0.0.1.0',
    'category': 'CRM',
    'summary': 'AI-powered Lead Scoring System for Odoo CRM',
    'author': 'Islam Abdelmoniem',
    'depends': ['base','crm'],
    'data': [

        'views/lead_views.xml',
        # 'views/lead_views.xml',  # Add any views or modifications here
    ],

    'external_dependencies': {
        'python': ['joblib', 'pandas', 'scikit-learn'],
    },
    'installable': True,
    'application': True,
}

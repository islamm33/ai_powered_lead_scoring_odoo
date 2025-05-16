# import sys
# sys.path.append(r"C:\Program Files\Odoo 18.0.20250413\server")

from odoo import models, fields, api, _
from odoo.exceptions import UserError
import json
import csv
import base64
from io import StringIO
import os
import joblib
import pandas as pd
from datetime import datetime
from odoo import models, fields, api, _
from odoo.exceptions import UserError
import logging
import random


_logger = logging.getLogger(__name__)




_logger = logging.getLogger(__name__)


class CrmLead(models.Model):
    _inherit = 'crm.lead'

    email_provided = fields.Boolean(
        string="Email Provided",
        compute='_compute_email_provided',
        store=True  # Optional: store the computed value for faster access
    )

    @api.depends('email_from')
    def _compute_email_provided(self):
        for rec in self:
            rec.email_provided = bool(rec.email_from)
    # Custom fields for additional features
    x_specialization = fields.Selection(
        selection=[
            ('select', 'Select'),
            ('finance', 'Finance Management'),
            ('hr', 'Human Resource Management'),
            ('marketing', 'Marketing Management'),
            ('operations', 'Operations Management'),
            ('business', 'Business Administration'),
            ('it', 'IT Projects Management'),
            ('supply_chain', 'Supply Chain Management'),
            ('banking', 'Banking, Investment And Insurance'),
            ('media', 'Media and Advertising')
        ],
        string="Specialization",
        default='select'
    )
    x_through_recommendations = fields.Boolean("Through Recommendations")
    test = fields.Boolean("test")



    # AI Tracking Fields
    ai_score = fields.Float("AI Conversion Score", digits=(3, 2), readonly=True)
    last_ai_update = fields.Datetime("Last AI Scoring", readonly=True)
    ai_model_version = fields.Char("AI Model Version", readonly=True)
    ai_features_json = fields.Text("Features Used", readonly=True)
    # Training Metrics Fields
    training_score = fields.Float(
        string='Training Accuracy',
        digits=(3, 2)
    )
    training_progress = fields.Float(
        string='Training Progress (%)',
        compute='_compute_training_progress'
    )
    last_training_update = fields.Datetime()

    def _compute_training_progress(self):
        for record in self:
            record.training_progress = record.training_score * 100

    def _get_model_path(self):
        """Get path to the trained model file"""
        return os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data',
            'calibrated_rf_odoo.joblib'
        )


    def _prepare_lead_features(self):
        """Convert Odoo lead to model input format"""

        def get_last_activity(lead):
            # Inner function to compute last activity
            last_act = lead.activity_ids.sorted('date_done', reverse=True)[:1]
            return last_act.activity_type_id.name if last_act else "none"

        # Convert tags to pipe-separated string
        tags = '|'.join(self.tag_ids.mapped('name')) if self.tag_ids else 'none'

        return {
            'Lead Source': self.source_id.name or 'unknown',
            'Do Not Email': 'no' if self.email_provided else 'yes',
            'Do Not Call': 'yes' if self.phone_sanitized_blacklisted else 'no',
            'Country': self.country_id.name or 'unknown',
            'City': self.city or 'unknown',
            'Specialization': self.x_specialization or 'Select',
            'Through Recommendations': 'yes' if self.x_through_recommendations else 'no',
            'Job Position': self.function or 'unknown',  # Comma added here
            'Last Notable Activity': get_last_activity(self),   # Fixed call
            'Tags': tags
        }

    def _get_pipeline_path(self):
        """
        Returns the absolute path to the saved trained pipeline.
        Adjust this path as needed.
        """
        # Example: if the pipeline is stored in the module's data folder.
        # Use a raw string (or double backslashes) to avoid escape issues:
        return r"C:\Program Files\Odoo 18.0.20250413\custom_addons\lead_scoring_ai\data\trained_pipelinex.joblib"

        # Add this method to your existing CrmLead class
    def generate_probability_based_on_job_position(self):
        """Generate realistic probability based on job position hierarchy"""
        job_titles_map = {
            5: ['CEO', 'CFO', 'CTO', 'COO', 'Chief Innovation Officer'],
            4: ['Vice President', 'Director of Operations', 'Head of Marketing', 'Senior Strategy Director'],
            3: ['Project Manager', 'Marketing Manager', 'Team Lead', 'Operations Manager'],
            2: ['Data Analyst', 'Software Engineer', 'Business Consultant', 'Technical Specialist'],
            1: ['Intern', 'Junior Developer', 'Assistant Coordinator', 'Associate Analyst']
        }

        # Default probability range for unknown positions
        current_level = 0
        current_job = (self.function or '').strip().lower()

        # Find the level for current job title
        for level, titles in job_titles_map.items():
            if any(title.lower() in current_job for title in titles):
                current_level = level
                break

        # Define probability ranges for each level
        probability_ranges = {
            5: (0.70, 0.90),  # C-Suite
            4: (0.60, 0.80),  # Senior Leadership
            3: (0.45, 0.65),  # Middle Management
            2: (0.30, 0.50),  # Professional Staff
            1: (0.15, 0.35),  # Junior Roles
            0: (0.10, 0.30)  # Unknown/Other
        }

        min_prob, max_prob = probability_ranges[current_level]
        return round(random.uniform(min_prob, max_prob), 2)

    # Modify your existing action_score_lead method
    def action_score_lead(self):
        """Generate realistic probability score based on position"""
        self.ensure_one()
        try:
            # Generate probability based on job position
            probability = self.generate_probability_based_on_job_position()

            # Write results
            self.write({
                'ai_score': probability * 100,
                'last_ai_update': fields.Datetime.now(),
                'ai_model_version': 'v1.2 (Position-based)',
                'ai_features_json': json.dumps({
                    'job_position': self.function,
                    'logic': 'position_hierarchy_based'
                })
            })

        except Exception as e:
            _logger.error("Scoring failed: %s", str(e), exc_info=True)
            raise UserError(_("Scoring failed. Please try again."))


   # def action_score_lead(self):
        """Score a lead using the trained pipeline with comprehensive validation"""
        # self.ensure_one()
        # try:
        #     # 1. Pipeline Loading with Validation
        #     pipeline_path = self._get_pipeline_path()
        #     if not os.path.exists(pipeline_path):
        #         raise UserError(_("Trained pipeline not found. Please train the model first."))
        #
        #     pipeline_data = joblib.load(pipeline_path)
        #     if not all(k in pipeline_data for k in ['pipeline', 'selected_columns']):
        #         raise UserError(_("Invalid pipeline format. Please retrain the model."))
        #
        #     pipeline = pipeline_data['pipeline']
        #     required_columns = pipeline_data['selected_columns']
        #     _logger.debug("Loaded pipeline expecting columns: %s", required_columns)
        #
        #     # 2. Feature Preparation with Fallbacks
        #     features = self._prepare_lead_features()
        #     _logger.debug("Raw features before alignment: %s", features)
        #
        #     # 3. Feature Alignment with Robust Handling
        #     features_df = pd.DataFrame([features])
        #
        #     # Handle missing columns
        #     missing_cols = set(required_columns) - set(features_df.columns)
        #     if missing_cols:
        #         _logger.warning("Adding missing columns with default values: %s", missing_cols)
        #         for col in missing_cols:
        #             features_df[col] = 0  # Numeric default
        #             if col in pipeline.named_steps['preprocessing'].categorical_columns:
        #                 features_df[col] = 'OTHER'  # Categorical default
        #
        #     # Ensure correct column order
        #     features_df = features_df.reindex(columns=required_columns)
        #
        #     # 4. Categorical Value Validation
        #     if hasattr(pipeline, 'named_steps') and 'preprocessing' in pipeline.named_steps:
        #         preprocessor = pipeline.named_steps['preprocessing']
        #         if hasattr(preprocessor, 'one_hot_enc') and hasattr(preprocessor.one_hot_enc, 'categories_'):
        #             for i, col in enumerate(preprocessor.categorical_columns):
        #                 if col in features_df:
        #                     valid_cats = preprocessor.one_hot_enc.categories_[i]
        #                     features_df[col] = features_df[col].apply(
        #                         lambda x: x if x in valid_cats else 'UNKNOWN'
        #                     )
        #
        #     _logger.info("Final features for prediction:\n%s", features_df)
        #
        #     # 5. Prediction with Validation
        #     if len(features_df) != 1:
        #         raise UserError(_("Feature preparation failed. Expected exactly one record."))
        #
        #     probability = pipeline.predict_proba(features_df)[0][1]
        #
        #     # 6. Result Storage
        #     self.write({
        #         'ai_score': probability,
        #         'last_ai_update': fields.Datetime.now(),
        #         'ai_model_version': pipeline_data.get('version', 'v1.0'),
        #         'ai_features_json': json.dumps({
        #             'input_features': features,
        #             'processed_features': features_df.iloc[0].to_dict()
        #         })
        #     })
        #
        # except ValueError as e:
        #     msg = f"Feature validation error: {str(e)}"
        #     _logger.error(msg, exc_info=True)
        #     raise UserError(_(msg))
        #
        # except Exception as e:
        #     _logger.error("Scoring failed for lead %s: %s", self.id, str(e), exc_info=True)
        #     raise UserError(_("Scoring failed. Details in server logs."))

    def export_leads_to_csv(self):
        """
        Exports each lead's data to CSV and creates an attachment for download.
        """
        leads = self.search([])

        # Create an in-memory CSV file
        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)

        # Define the header row (without the 'Test' column)
        header = [
            'Name',
            'Lead Source',

            'Do Not Email',
            'Do Not Call',
            'Country',
            'City',
            'Specialization',
            'Through Recommendations',
            'Job Position',
            'Last Notable Activity',
            'Tags'
        ]
        writer.writerow(header)

        def get_last_activity(lead):
            last_act = lead.activity_ids.sorted('date_done', reverse=True)[:1]
            return last_act.activity_type_id.name if last_act else "none"

        # Iterate over each lead and write a row of values
        for lead in leads:
            row = [
                lead.name or '',
                lead.source_id.name or 'unknown',
                'no' if lead.email_provided else 'yes',
                'yes' if lead.phone_sanitized_blacklisted else 'no',
                lead.country_id.name if lead.country_id else 'unknown',
                lead.city or 'unknown',
                lead.x_specialization or 'Select',  # Custom field
                'yes' if lead.x_through_recommendations else 'no',  # Custom field
                lead.function or 'unknown',  # Fetched field for Job Position
                get_last_activity(lead),  # Fetched field for Last Notable Activity
                '|'.join(lead.tag_ids.mapped('name')) if lead.tag_ids else 'none'
            ]
            writer.writerow(row)

        csv_content = csv_buffer.getvalue()
        csv_buffer.close()

        attachment = self.env['ir.attachment'].create({
            'name': 'exported_leads.csv',
            'datas': base64.b64encode(csv_content.encode('utf-8')),
            'res_model': 'crm.lead',
            'res_id': 0,
            'type': 'binary',
            'mimetype': 'text/csv'
        })

        return {
            'type': 'ir.actions.act_url',
            'url': '/web/content/%s?download=true' % attachment.id,
            'target': 'new',
        }

    # import joblib
    # import pandas as pd
    #
    # # Path to your saved pipeline joblib file; update this to your actual path.
    # pipeline_path = r'C:\Program Files\Odoo 18.0.20250413\custom_addons\lead_scoring_ai\data\trained_pipelinex.joblib'
    #
    # # Load your saved pipeline from the joblib file.
    # pipeline = joblib.load(pipeline_path)
    #
    # # Create a test record, ensuring you include all the columns your pipeline was trained on.
    # data = {
    #     'Lead Source': 'Direct Traffic',  # Use an appropriate test value
    #     'Do Not Email': 'No',  # Must be consistent with training values (e.g., 'yes'/'no' or true/false)
    #     'Do Not Call': 'No',  # Adjust as needed
    #     'Country': 'India',  # Example value
    #     'City': 'Mumbai',  # Example value
    #     'Specialization': 'Select',  # Example value; match what you expect in training
    #     'Through Recommendations': 'No',  # Adjust as needed, for instance
    #     'Last Notable Activity': 'Email Opened',  # Example text
    #     'Tags': 'Will revert after reading the email'  # Example tags, if these are what you used during training
    # }
    #
    # # Convert the data dictionary into a DataFrame with one record.
    # test_df = pd.DataFrame([data])
    #
    # # Output the input record for verification.
    # print("Input Record:")
    # print(test_df)
    #
    # # Use the pipeline to predict probabilities.
    # # For binary classification, the output will be an array with probabilities for class 0 and class 1.
    # probabilities = pipeline.predict_proba(test_df)
    #
    # print("\nPredicted Probabilities:")
    # print(probabilities)
    #
    # # Optionally: If you want to see the predicted class as well:
    # predicted_class = pipeline.predict(test_df)
    # print("\nPredicted Class:")
    # print(predicted_class)

    def action_train_model(self):
        """Train model and update progress"""
        try:
            from .train_pipeline import run_training
            score = run_training()

            # Update global training data
            self.env['ir.config_parameter'].sudo().set_param(
                'lead_scoring.training_score',
                score
            )
            self.env['ir.config_parameter'].sudo().set_param(
                'lead_scoring.last_training_update',
                fields.Datetime.now()
            )

            # Force recompute of training_progress on all records
            self.search([]).write({'training_score': score})

            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': 'Training Complete',
                    'message': f'Model accuracy: {score:.2%}',
                    'sticky': False,
                }
            }
        except FileNotFoundError as e:
            _logger.error("Missing training file: %s", str(e))
            return self._training_error(f"Training data not found: {str(e)}")

        except Exception as e:
            _logger.error("Training failed: %s", str(e), exc_info=True)
            return self._training_error(f"Training failed: {str(e)}")

    def _training_error(self, message):
        """Helper for error notifications"""
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': 'Training Error',
                'message': message,
                'type': 'danger',
                'sticky': True,
            }
        }

    # @api.model
    # def _cron_batch_score_leads(self, batch_size=200):
    #     """Batch scoring for recent/updated leads"""
    #     try:
    #         model = joblib.load(self._get_model_path())
    #         leads = self.search([
    #                                 ('stage_id.is_won', '=', False),
    #                                 '|',
    #                                 ('last_ai_update', '=', False),
    #                                 ('write_date', '>', fields.Datetime.to_string(
    #                                     fields.Datetime.from_string(self.last_ai_update)
    #                                 )
    #                                  ], limit=batch_size)
    #
    #                             for lead in leads:
    #         try:
    #             features = lead._prepare_lead_features()
    #             lead.write({
    #                 'ai_score': model.predict_proba([features])[0][1],
    #                 'last_ai_update': fields.Datetime.now(),
    #                 'ai_model_version': datetime.now().strftime('v%Y%m%d'),
    #                 'ai_features_json': str(features)
    #             })
    #         except Exception as e:
    #             _logger.error(f"Failed scoring lead {lead.id}: {str(e)}")
    #             continue
    #
    #     except Exception as e:
    #         _logger.error(f"Batch scoring failed: {str(e)}", exc_info=True)
    #         # Consider adding email notification to admin
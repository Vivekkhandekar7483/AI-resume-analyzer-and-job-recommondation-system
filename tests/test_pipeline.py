import os
import unittest
import pandas as pd

from resume_parser import extract_text
from text_cleaner import clean_text
from skill_extractor import load_skills, extract_skills
from job_matcher import score_roles, predict_ml_role
from roadmap_generator import make_roadmap
from report_generator import generate_text_report


class TestResumeAnalyzerPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.skills_path = os.path.join(cls.base_dir, "data", "skill_dictionary.csv")
        cls.roles_path = os.path.join(cls.base_dir, "data", "job_roles.csv")
        cls.skills = load_skills(cls.skills_path)
        cls.roles_df = pd.read_csv(cls.roles_path)

    def test_text_cleaner(self):
        raw_text = (
            "Name: Jane Doe\n"
            "Email: jane@example.com\n"
            "Phone: +1-555-0199\n"
            "Gender: Female, DOB: 1998-05-12\n"
            "Skills: Proficient in Python, C++, C#, .NET, HTML/CSS, React, SQL, and Docker!"
        )
        cleaned = clean_text(raw_text)
        self.assertNotIn("jane@example.com", cleaned)
        self.assertNotIn("female", cleaned)
        self.assertIn("python", cleaned)
        self.assertIn("c++", cleaned)
        self.assertIn("c#", cleaned)
        self.assertIn(".net", cleaned)
        self.assertIn("html/css", cleaned)

    def test_skill_extractor(self):
        sample_resume = "experienced software engineer proficient in python, scikit-learn, fastapi, docker, and sql."
        found = extract_skills(sample_resume, self.skills)
        self.assertIn("Python", found)
        self.assertIn("scikit-learn", found)
        self.assertIn("FastAPI", found)
        self.assertIn("Docker", found)
        self.assertIn("SQL", found)

    def test_job_matcher(self):
        sample_resume = "python pandas sql excel power bi statistics data visualization"
        found = extract_skills(sample_resume, self.skills)
        results = score_roles(sample_resume, found, self.roles_df)

        self.assertFalse(results.empty)
        top_role = results.iloc[0]
        self.assertEqual(top_role["Role"], "Data Analyst")
        self.assertGreaterEqual(top_role["Score"], 70.0)
        self.assertIn("Python", top_role["Matched"])

    def test_ml_predictor(self):
        sample_java_resume = "experienced senior java developer spring boot microservices hibernate REST API Oracle SQL git maven jenkins"
        ml_pred = predict_ml_role(sample_java_resume)
        self.assertTrue(ml_pred["available"])
        self.assertEqual(ml_pred["top_role"], "Java Developer")
        self.assertGreater(ml_pred["confidence"], 0.0)


    def test_roadmap_generator(self):
        missing = ["FastAPI", "Docker", "MLflow"]
        roadmap = make_roadmap(missing, self.skills)
        self.assertTrue(any("FastAPI" in item for item in roadmap))
        self.assertTrue(any("Docker" in item for item in roadmap))
        self.assertTrue(any("MLflow" in item for item in roadmap))

    def test_report_generator(self):
        report = generate_text_report(
            target_role="Machine Learning Engineer",
            score=85.0,
            matched=["Python", "scikit-learn"],
            missing=["FastAPI", "Docker"],
            top_roles_df=self.roles_df,
            roadmap=["Week 1: FastAPI", "Week 2: Docker"]
        )
        self.assertIn("AI RESUME ANALYZER", report)
        self.assertIn("Machine Learning Engineer", report)
        self.assertIn("Python", report)

    def test_docx_parser(self):
        sample_docx = os.path.join(self.base_dir, "sample_resumes", "Resume_Data_Analyst.docx")
        if os.path.exists(sample_docx):
            with open(sample_docx, "rb") as f:
                content = f.read()
            text = extract_text(content, "Resume_Data_Analyst.docx")
            cleaned = clean_text(text)
            self.assertIn("alex rivera", cleaned)
            self.assertIn("python", cleaned)


if __name__ == "__main__":
    unittest.main()

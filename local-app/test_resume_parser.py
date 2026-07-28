import sys
import os

# Ensure local-app directory is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

from modules.interview.agents.resume_intelligence import ResumeIntelligenceAgent

def test_resume_parser():
    agent = ResumeIntelligenceAgent()
    # Mock LLM response to test static block flipping parser directly
    agent.llm_parse_resume = lambda text: {}

    print("=" * 70)
    print("TEST 1: Standard ATS Resume")
    print("=" * 70)

    ats_resume = """
    ALEXANDER RIVERS
    alex.rivers@email.com | +1 (555) 234-5678 | San Francisco, CA

    PROFESSIONAL SUMMARY
    Senior Backend Engineer with 5+ years of experience in Python, FastAPI, and Docker.

    WORK EXPERIENCE
    Senior Backend Developer | TechCorp Inc. (2022 - Present)
    - Engineered microservices using Python, FastAPI, and PostgreSQL.
    - Optimized database query performance by 40%.

    Software Engineer | DataScale Systems (2019 - 2022)
    - Developed real-time streaming pipelines using Python and Redis.

    TECHNICAL SKILLS
    Languages: Python, SQL, C++
    Frameworks: FastAPI, React
    Tools & Cloud: Docker, AWS, PostgreSQL, Git

    EDUCATION
    Bachelor of Technology in Computer Science | Stanford University (2015 - 2019)
    GPA: 3.8 / 4.0

    PROJECTS
    Distributed KV Store: Built a high-performance in-memory key-value store using C++ and Docker.
    """

    res1 = agent.extract_profile(ats_resume)
    print(f"Extracted Name: {res1['full_name']}")
    print(f"Extracted Email: {res1['email']}")
    print(f"Extracted Phone: {res1['phone']}")
    print(f"Work History Count: {len(res1['work_history'])}")
    print(f"Education Count: {len(res1['education'])}")
    print(f"Projects Count: {len(res1['projects'])}")
    print(f"Projects: {res1['projects']}")

    assert res1['full_name'] == "Alexander Rivers", f"Expected Alexander Rivers, got {res1['full_name']}"
    assert res1['email'] == "alex.rivers@email.com"
    assert len(res1['work_history']) >= 1
    assert len(res1['education']) >= 1
    assert len(res1['projects']) >= 1


    print("\n" + "=" * 70)
    print("TEST 2: Non-ATS Resume with Education First & Miscategorized Internships/Projects under Education")
    print("=" * 70)

    scrambled_resume = """
    Jane von Neumann
    j.neumann@mit.edu | 617-555-0199

    EDUCATION & ACADEMICS
    B.S. in Computer Science and Electrical Engineering
    Massachusetts Institute of Technology (2020 - 2024)
    GPA: 3.9 / 4.0

    Software Engineer Intern at Acme Robotics Corp (Jun 2023 - Sep 2023)
    - Developed computer vision algorithms using PyTorch and OpenCV for autonomous rover navigation.
    - Designed real-time sensor fusion pipeline for RGB-Thermal cameras.

    1st Place Winner - MIT Hackathon 2023
    - Awarded best AI hardware integration for autonomous gesture tracking.

    THINGS I'VE BUILT
    Smart Drone Assistant: Developed a real-time gesture control system using OpenCV, Python and TensorFlow. https://github.com/jane/drone-ai

    AI Speech Synthesizer: Built a real-time web application with React and FastAPI for speech cloned voice conversion.
    """

    res2 = agent.extract_profile(scrambled_resume)
    print(f"Extracted Name: {res2['full_name']}")
    print(f"Extracted Email: {res2['email']}")
    print(f"Work History (Flipped from Education): {res2['work_history']}")
    print(f"Education: {res2['education']}")
    print(f"Projects: {res2['projects']}")
    print(f"Achievements: {res2['achievements']}")

    assert "Jane" in res2['full_name'], f"Expected Jane in name, got {res2['full_name']}"
    assert res2['email'] == "j.neumann@mit.edu"
    assert len(res2['work_history']) >= 1, "Internship under Education should be flipped to work_history!"
    assert len(res2['education']) >= 1
    assert len(res2['projects']) >= 2, f"Expected at least 2 projects, got {len(res2['projects'])}"

    print("\n" + "=" * 70)
    print("ALL RESUME PARSER TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_resume_parser()

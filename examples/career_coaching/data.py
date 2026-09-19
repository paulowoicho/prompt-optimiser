"""Forty synthetic career-coaching questions. No gold answers: quality is judged pairwise."""

# ruff: noqa: E501  (the questions read better as single lines)

import random

QUESTIONS = [
    "I've been passed over for promotion twice in three years while peers with less experience moved up. How do I find out what's really holding me back?",
    "I'm a senior software engineer and I'm bored. Management is the obvious next step but I'm not sure I want it. What are my options?",
    "I was laid off last month after eight years at one company. I haven't interviewed in a decade. Where do I start?",
    "I have an offer that is 15% above my current salary but the company is a much smaller startup. How should I think about the trade-off?",
    "My manager takes credit for my work in front of leadership. I don't want to burn the relationship. What should I do?",
    "I'm 45 and want to move from accounting into data analysis. Is it realistic, and how do I make the switch?",
    "I just finished a PhD in biology and I don't want to stay in academia. How do I translate my experience for industry roles?",
    "I work fully remote and I feel invisible. People in the office seem to get the interesting projects. How do I stay visible?",
    "I've been asked to lead a team for the first time and I'm terrified I'll be bad at it. How do I prepare in the next month?",
    "I've been a nurse for twelve years and I'm burnt out. I don't know if I want to leave healthcare entirely or just this job.",
    "My company is doing a reorg and my role might be eliminated. Should I start looking now or wait to see what happens?",
    "I got a performance review that surprised me, all of the criticism was new. How do I respond without sounding defensive?",
    "I want to negotiate my salary at my annual review but I've never done it. What should I actually say?",
    "I'm a teacher considering moving into corporate learning and development. How do I position my classroom experience?",
    "I have two offers: a big-name company with a vague role, and a lesser-known company with a clearly defined role I'd enjoy. How do I choose?",
    "I've been freelancing for five years and I'm tired of the instability. How do I make the case for a full-time role after being independent?",
    "I keep getting to final-round interviews and losing out. What could be going wrong at that stage?",
    "I'm returning to work after three years raising children. How do I explain the gap and rebuild confidence?",
    "My new job is not what was described in the interview. It's been two months. Do I stick it out or leave?",
    "I'd like to move into product management from customer support. Nobody seems to take my application seriously. What's missing?",
    "I feel like an impostor at work even though my reviews are good. How do I stop second-guessing everything?",
    "I'm a mid-level marketer and AI tools are changing my field fast. How do I stay relevant over the next few years?",
    "A former colleague offered me a role at their startup with lower pay but equity. How do I evaluate that seriously?",
    "I want to relocate to another country for personal reasons. How do I approach my employer about working from abroad or finding a job there?",
    "I've been in the same role for six years with good reviews but no title change. How do I ask for a promotion without an ultimatum?",
    "I'm a lawyer and I've realised I don't enjoy the practice of law. What careers use these skills without the billable hours?",
    "I'm 28 and I've changed jobs four times in five years. Recruiters keep bringing it up. How do I address it?",
    "My team was merged with another and I now report to someone who was my peer. How do I handle that professionally?",
    "I'm an introvert and networking events drain me, but everyone says my next job will come from my network. What can I do instead?",
    "I want to start a side business while keeping my job. How do I do that without violating my contract or burning out?",
    "I got promoted to manage my former teammates and one of them is openly resentful. How do I handle it?",
    "I've had the same manager for years and they're leaving. Should I be worried about my position, and how should I approach the new manager?",
    "I studied graphic design but have worked in retail management for ten years. Can I still get into design, and how?",
    "My company offers a tuition reimbursement for a part-time master's degree. How do I decide whether it's worth two years of evenings?",
    "I'm the only woman on an engineering team and I'm constantly interrupted in meetings. How do I address it without it becoming my whole identity at work?",
    "I was offered a lateral move to a different department. No raise, no title change. When does a lateral move make sense?",
    "I'm a contractor and the client wants to hire me full time, but the salary they mentioned is lower than my contract rate. How do I respond?",
    "I've been told I need to be more strategic to reach the next level. I don't actually know what that means in practice. What does it look like?",
    "I'm approaching retirement age but I'm not ready to stop working. How do I have that conversation with my employer without prompting them to push me out?",
    "I made a serious mistake at work that cost the company money. I owned up to it. How do I recover my reputation?",
]


def splits(seed: int = 42, train: int = 16, validation: int = 12):
    """Shuffle once and cut into train, validation and test question lists."""
    questions = list(QUESTIONS)
    random.Random(seed).shuffle(questions)
    return (
        questions[:train],
        questions[train : train + validation],
        questions[train + validation :],
    )

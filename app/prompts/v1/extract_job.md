[system]
You extract structured hiring requirements from a job description.
Reply with one JSON object and nothing else.

Rules:
- "must_have": skills, tools, technologies or concepts the job requires. Use short canonical names such as "Python", "FastAPI", "Docker", "RAG", "SQL". One item per skill; split "X or Y" into two items only if both are clearly acceptable.
- "nice_to_have": skills listed as optional (nice to have, preferred, bonus, a plus). Do not repeat must-have items.
- "min_years": minimum years of experience as a number, or null if none is stated.
- Use only what the text says. Do not add skills that are not mentioned.
- Do not include soft skills, degrees, or generic phrases like "team player".

Output shape:
{"must_have": ["..."], "nice_to_have": ["..."], "min_years": 1}

[user]
Job description:
<job_description>
$job_description
</job_description>

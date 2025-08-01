import os
import json
import re
import requests
import openai
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from datetime import datetime
import concurrent.futures
from waitress import serve

load_dotenv()
app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GNB_TICKET_URL = "https://globalnoticeboard.com/admin/get_client_data_api.php"
openai.api_key = OPENAI_API_KEY


def clean_openai_json(raw_text):
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip(), flags=re.IGNORECASE)
    return cleaned.strip()


def call_openai(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            temperature=0,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": "You are a customer experience analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI Error: {str(e)}"


def calculate_response_delay(created_date_str, responded_content):
    if not responded_content or len(responded_content) < 10:
        return -1
    try:
        created_dt = datetime.strptime(created_date_str, "%d.%m.%y %H:%M")
        response_dt_str = responded_content.split(" ")[0]
        response_dt = datetime.strptime(response_dt_str, "%d.%m.%y")
        return max((response_dt - created_dt).days, 0)
    except Exception:
        return -1


def score_ticket_batch(batch, batch_index):
    ticket_texts = []
    for t in batch:
        ticket_texts.append(f"""
Ticket Number: {t["cnb_support_ticket_number"]}
Title: {t["cnb_support_ticket_title"]}
Priority: {t.get("cnb_support_ticket_priority", "Normal")}
Created: {t.get("cnb_created_datetime", "")}
Response: {t.get("responded_content_with_datetime", "")}
Response Delay: {t['response_delay']} days
""")

    ticket_batch_prompt = f"""
Evaluate each of the following support tickets. Score each from 0 to 10 based on:
- Sentiment
- Relationship tone
- Support quality
- Priority + Response time

📌 Priority Rules:
- Urgent: 2 days
- High: 3 days
- Normal: 5 days
- Low: 7 days

Penalize if delayed or no response. For each ticket, return:
- ticket_number
- ticket_score
- reason (short 2-line justification)

Return format: JSON array like:
[
  {{
    "ticket_number": "...",
    "ticket_score": number,
    "reason": "..."
  }},
  ...
]

Tickets:
{''.join(ticket_texts)}
"""

    try:
        raw = call_openai(ticket_batch_prompt)
        cleaned = clean_openai_json(raw)
        return json.loads(cleaned)
    except Exception:
        return [{
            "ticket_number": f"Batch-{batch_index + 1}",
            "ticket_score": "Error",
            "reason": "Failed to parse batch response"
        }]


@app.route("/analyzing", methods=["POST"])
def analyze_ticket():
    cnb_id = request.form.get("cnb_id")

    if not cnb_id:
        return jsonify({"error": "'cnb_id' required."}), 400

    try:
        # Fetch ticket data
        response = requests.post(GNB_TICKET_URL, data={"cnb_id": cnb_id})
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch ticket data"}), 500

        raw_text = response.text.strip()
        cleaned_text = re.sub(r'^<pre>|</pre>$', '', raw_text)
        data = json.loads(cleaned_text)

        cnb_title = data.get("cnb_title", "Unknown Client")

        total_tickets = 0
        book_training_count = 0
        no_response_count = 0
        valid_tickets = []

        for k, v in data.items():
            if not k.isdigit():
                continue
            total_tickets += 1
            title = v.get("cnb_support_ticket_title", "").strip().lower()
            response_content = v.get("responded_content_with_datetime", "")
            if any(bt in title for bt in ["book a training", "book training"]):
                book_training_count += 1
                continue
            if not response_content or not response_content.strip():
                no_response_count += 1
                continue
            v["response_delay"] = calculate_response_delay(v.get("cnb_created_datetime", ""), response_content)
            valid_tickets.append(v)

        # ---------------- PARALLEL BATCHING ----------------
        batch_size = 50
        batches = [valid_tickets[i:i + batch_size] for i in range(0, len(valid_tickets), batch_size)]
        ticket_score_data = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_batch = {
                executor.submit(score_ticket_batch, batch, i): batch
                for i, batch in enumerate(batches)
            }

            for future in concurrent.futures.as_completed(future_to_batch):
                result = future.result()
                batch = future_to_batch[future]
                for ticket_result in result:
                    ticket_number = ticket_result.get("ticket_number")
                    matching_ticket = next((t for t in batch if t["cnb_support_ticket_number"] == ticket_number), None)
                    if matching_ticket:
                        ticket_score_data.append({
                            "ticket_number": ticket_number,
                            "ticket_title": matching_ticket.get("cnb_support_ticket_title", ""),
                            "ticket_priority": matching_ticket.get("cnb_support_ticket_priority", ""),
                            "ticket_score": ticket_result.get("ticket_score"),
                            "reason": ticket_result.get("reason")
                        })
                    else:
                        ticket_score_data.append(ticket_result)

        # ---------------- OVERALL SCORE (BATCHED & SUMMARIZED) ----------------
        BATCH_SIZE_OVERALL = 100  # You can adjust this as needed

        # Split valid tickets into batches for overall score
        overall_batches = [valid_tickets[i:i + BATCH_SIZE_OVERALL] for i in range(0, len(valid_tickets), BATCH_SIZE_OVERALL)]
        overall_scores = []

        for batch in overall_batches:
            # Summarize each ticket as a single line (title + short content)
            summarized_tickets = [
                f"Title: {t['cnb_support_ticket_title']} | Message: {t['cnb_support_ticket_content'][:100]} | Response: {t['responded_content_with_datetime'][:100]}"
                for t in batch
            ]
            summary_input = "\n".join(summarized_tickets)

            overall_prompt = f"""
Analyze the following summarized support tickets. Provide an overall score (0-10) for this batch based on:
- Sentiment of client messages
- Relationship tone
- Quality of responses

⚠️ The same input must always return the same score. Do not vary your scoring unless the ticket content changes.

Client: {cnb_title}

Tickets:
{summary_input}

Return format as JSON:
{{ "overall_score": number }}
Note: The score must be an integer.
"""
            overall_raw = call_openai(overall_prompt)
            match = re.search(r'\{[\s\S]*?\}', overall_raw)
            if not match:
                # Optionally, log or print the problematic response for debugging
                overall_scores.append(0)  # Or skip/continue if you prefer
                continue
            overall_clean = match.group(0)
            try:
                batch_score = json.loads(overall_clean)["overall_score"]
                overall_scores.append(int(batch_score))
            except Exception:
                overall_scores.append(0)  # Or skip/continue if you prefer
                continue

        # Average the batch scores for the final overall score
        if overall_scores:
            final_overall_score = round(sum(overall_scores) / len(overall_scores))
        else:
            final_overall_score = 0

        return jsonify({
            "cnb_id": cnb_id,
            "client_name": cnb_title,
            "total_tickets": total_tickets,
            "book_training_tickets": book_training_count,
            "tickets_without_response": no_response_count,
            "overall_score": final_overall_score,
            "ticket_details": ticket_score_data
        })

    except Exception as e:
        return jsonify({"error": "Unexpected error", "details": str(e)}), 500


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)
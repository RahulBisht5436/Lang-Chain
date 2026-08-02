import streamlit as st
from langchain_core.prompts import PromptTemplate

from groq_llm import llm


travel_prompt = PromptTemplate(
    input_variables=[
        "destination",
        "origin",
        "days",
        "travelers",
        "budget",
        "currency",
        "interests",
        "pace",
        "accommodation",
        "language",
        "requirements",
    ],
    template="""
You are an expert travel planner. Create a practical, personalized itinerary
using the following details:

- Destination: {destination}
- Traveling from: {origin}
- Trip duration: {days} days
- Number of travelers: {travelers}
- Total budget: {budget} {currency}
- Interests: {interests}
- Preferred pace: {pace}
- Accommodation preference: {accommodation}
- Preferred response language: {language}
- Dietary, accessibility, or other requirements: {requirements}

Write the guide in {language} and organize it with these sections:

# Welcome to {destination}
Give a short overview of the destination and the best way to enjoy this trip.

# Before You Go
Mention useful preparation, local customs, weather considerations, safety,
payment methods, connectivity, and what to pack. Do not invent exact current
prices, opening hours, visa rules, or live events; clearly advise the traveler
to verify details that may change.

# Day-by-Day Itinerary
Create a realistic plan for every day. For each day include:
- Morning, afternoon, and evening activities
- Approximate time needed
- Sensible travel order to reduce unnecessary transport
- One local food recommendation
- A lower-cost or indoor alternative when useful

# Estimated Budget
Provide a clear estimated breakdown for accommodation, food, local transport,
attractions, and a contingency amount. Keep the total near {budget} {currency}
and explain if the requested budget is unrealistic.

# Local Essentials
Include transportation advice, useful phrases, etiquette, common tourist
mistakes, emergency guidance, and practical money-saving tips.

# Final Checklist
End with a concise checklist and one memorable travel tip.

Make the itinerary specific and useful, not generic. Use readable Markdown.
"""
)

st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")

st.title("✈️ AI Travel Planner")
st.caption("Build a personalized itinerary, budget, and local travel guide.")

with st.form("travel_planner"):
    col1, col2 = st.columns(2)

    with col1:
        destination = st.text_input(
            "Destination",
            placeholder="e.g. Kyoto, Japan",
        )
        origin = st.text_input(
            "Traveling from *",
            placeholder="e.g. New Delhi, India",
        )
        days = st.number_input(
            "Trip duration (days)",
            min_value=1,
            max_value=30,
            value=5,
        )
        travelers = st.number_input(
            "Number of travelers",
            min_value=1,
            max_value=20,
            value=1,
        )

    with col2:
        budget = st.number_input(
            "Total trip budget",
            min_value=0,
            value=1000,
            step=100,
        )
        currency = st.selectbox(
            "Currency",
            ["USD", "INR", "EUR", "GBP", "JPY", "AUD", "CAD"],
        )
        pace = st.select_slider(
            "Travel pace",
            options=["Relaxed", "Balanced", "Fast-paced"],
            value="Balanced",
        )
        accommodation = st.selectbox(
            "Accommodation",
            ["Budget hostel", "Hotel", "Resort", "Apartment", "No preference"],
        )

    interests = st.multiselect(
        "Interests",
        [
            "Local culture",
            "History",
            "Food",
            "Nature",
            "Adventure",
            "Art and museums",
            "Shopping",
            "Nightlife",
            "Photography",
            "Family activities",
            "Relaxation",
        ],
        default=["Food", "Local culture"],
    )

    language = st.text_input("Guide language", value="English")
    requirements = st.text_area(
        "Special requirements",
        placeholder=(
            "Dietary needs, accessibility requirements, children, preferred "
            "activities, places to avoid, or anything else."
        ),
    )

    generate = st.form_submit_button(
        "Generate My Travel Guide",
        type="primary",
        use_container_width=True,
    )

if generate:
    if not origin.strip():
        st.warning("Please enter where you are traveling from.")
    else:
        destination_value = destination.strip() or "a destination that fits this trip"
        budget_value = budget if budget > 0 else "flexible / not specified"

        prompt = travel_prompt.format(
            destination=destination_value,
            origin=origin.strip(),
            days=days,
            travelers=travelers,
            budget=budget_value,
            currency=currency if budget > 0 else "",
            interests=", ".join(interests) or "General sightseeing",
            pace=pace,
            accommodation=accommodation,
            language=language.strip() or "English",
            requirements=requirements.strip() or "None",
        )

        spinner_label = (
            f"Planning your {days}-day trip"
            + (f" to {destination.strip()}" if destination.strip() else "")
            + "..."
        )

        with st.spinner(spinner_label):
            try:
                response = llm.invoke(prompt)
                st.success("Your travel guide is ready!")
                st.markdown(response.content)
                file_name = (
                    f"{destination.strip()}_travel_guide.md"
                    if destination.strip()
                    else "travel_guide.md"
                )
                st.download_button(
                    "Download Guide",
                    data=response.content,
                    file_name=file_name,
                    mime="text/markdown",
                )
            except Exception as error:
                st.error(f"Could not generate the travel guide: {error}")
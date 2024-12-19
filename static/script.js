// Toggle Sections Based on Dropdown Selection
document.getElementById("actionSelector").addEventListener("change", function () {
    const action = this.value;
    document.getElementById("askQuestionSection").style.display =
        action === "ask" ? "block" : "none";
    document.getElementById("refineDatabaseSection").style.display =
        action === "refine" ? "block" : "none";
});

// Phase 1: Ask a Question
document.getElementById("questionForm").addEventListener("submit", async function (e) {
    e.preventDefault();
    const question = document.getElementById("questionInput").value.trim();

    if (!question) {
        alert("Question cannot be empty!");
        return;
    }

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ question }),
        });

        const data = await response.json();
        if (response.ok) {
            document.getElementById("dbAnswer").innerText = data.database || "No match found.";
            document.getElementById("answers").style.display = "block";
        } else {
            alert(data.error || "An error occurred.");
        }
    } catch (error) {
        console.error("Error in Phase 1:", error);
        alert("Unexpected error occurred.");
    }
});

// Phase 2: Refining Questions
let dbQuestions = [];
let currentIndex = 0;

// Fetch All Questions
async function fetchQuestions() {
    try {
        const response = await fetch("/questions");
        const data = await response.json();
        if (response.ok) {
            dbQuestions = data.questions;
        } else {
            alert("Failed to fetch questions.");
        }
    } catch (error) {
        console.error("Error fetching questions:", error);
        alert("Unexpected error occurred.");
    }
}

document.getElementById("loadQuestionButton").addEventListener("click", function () {
    const questionNumber = parseInt(document.getElementById("questionNumberInput").value.trim(), 10);

    if (!dbQuestions.length) {
        alert("Questions not loaded yet.");
        return;
    }

    console.log(`Question Number Entered: ${questionNumber}`); // Debug log

    if (isNaN(questionNumber) || questionNumber <= 0 || questionNumber > dbQuestions.length) {
        console.log("Invalid question number entered."); // Debug log
        alert(`Invalid question number. Please enter a number between 1 and ${dbQuestions.length}`);
        return;
    }

    currentIndex = questionNumber - 1; // Zero-based index
    displayCurrentQuestion();
    document.getElementById("questionAnswerSection").style.display = "block";
});



// Display Current Question
function displayCurrentQuestion() {
    const questionObj = dbQuestions[currentIndex];
    const questionNumber = currentIndex + 1; // Convert to 1-based index

    document.getElementById("currentQuestion").innerHTML = `<strong>Question ${questionNumber}:</strong> ${questionObj.question}`;
    document.getElementById("currentAnswer").innerText = questionObj.answer;
    document.getElementById("refineCustomAnswer").value = "";

    console.log(`Displaying Question ${questionNumber}: ${questionObj.question}`); // Debug log
}



document.getElementById("submitAnswer").addEventListener("click", async function () {
    const question = document.getElementById("questionInput").value.trim();
    const newAnswer = document.getElementById("customAnswer").value.trim() || document.getElementById("dbAnswer").innerText.trim();

    if (!newAnswer) {
        alert("Please provide a valid answer.");
        return;
    }

    try {
        const response = await fetch("/add", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, answer: newAnswer }),
        });

        if (response.ok) {
            alert("Answer submitted successfully.");
            document.getElementById("questionForm").reset();
            document.getElementById("answers").style.display = "none";
        } else {
            alert("Failed to submit the answer.");
        }
    } catch (error) {
        console.error("Error in submission:", error);
    }
});


// Handle Refinement Submission
document.getElementById("submitRefine").addEventListener("click", async function () {
    const selectedOption = document.querySelector(
        'input[name="refineValidation"]:checked'
    );

    if (!selectedOption) {
        alert("Please validate the answer or provide a new one.");
        return;
    }

    const question = dbQuestions[currentIndex].question;
    let newAnswer;

    if (selectedOption.value === "custom") {
        newAnswer = document.getElementById("refineCustomAnswer").value.trim();
        if (!newAnswer) {
            alert("Please provide a custom answer.");
            return;
        }
    } else {
        newAnswer = document.getElementById("currentAnswer").innerText.trim();
    }

    try {
        const response = await fetch("/add", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ question, answer: newAnswer }),
        });

        if (response.ok) {
            alert("Answer updated successfully.");
            currentIndex++;
            if (currentIndex < dbQuestions.length) {
                displayCurrentQuestion();
            } else {
                alert("No more questions to refine.");
                document.getElementById("questionAnswerSection").style.display = "none";
            }
        } else {
            alert("Failed to update answer.");
        }
    } catch (error) {
        console.error("Error updating question:", error);
        alert("Unexpected error occurred.");
    }
});

// Initialize Questions for Refinement
document.getElementById("actionSelector").addEventListener("change", function () {
    if (this.value === "refine") {
        fetchQuestions();
    }
});

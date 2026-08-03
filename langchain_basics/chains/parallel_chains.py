import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.groq_llm import llm
from langchain_core import output_parsers
from langchain_core.prompts import PromptTemplate
from langchain_core.globals import set_debug
from langchain_core.output_parsers import StrOutputParser


textInformation = """

Quantum Computing
Definition

Quantum Computing is a new type of computing that uses the principles of quantum mechanics to process information. Unlike traditional computers, which store information as bits (0 or 1), quantum computers use quantum bits (qubits) that can exist in multiple states simultaneously through superposition.

This allows quantum computers to solve certain types of problems much faster than classical computers.

Introduction

For over 70 years, computers have become faster by making transistors smaller. However, there is a physical limit to how small transistors can become.

Quantum computing is an alternative approach.

Instead of depending on electrical circuits, it uses the laws of quantum physics to perform computations.

Researchers believe that quantum computers could solve problems that would take today's fastest supercomputers millions of years.

History
1980s

Richard Feynman proposed that quantum systems should be simulated using quantum computers because classical computers struggle to simulate quantum mechanics.

1985

David Deutsch introduced the concept of a universal quantum computer.

1994

Peter Shor developed Shor's Algorithm, showing that a quantum computer could factor very large numbers exponentially faster than classical computers.

This threatened modern cryptography.

1996

Lov Grover introduced Grover's Search Algorithm, providing quadratic speedup for searching unsorted databases.

Present

Companies like IBM, Google, Microsoft, Intel, Amazon, and IonQ are actively building quantum computers.

Basic Concepts
1. Bit

A classical computer stores information using bits.

A bit can only be

0
or
1

Example

10101011
2. Qubit

A qubit is the quantum version of a bit.

Instead of being only

0

or

1

it can be

0
1
Both simultaneously

This is called Superposition.

3. Superposition

Superposition allows a qubit to exist in multiple states at once.

Imagine a spinning coin.

A normal coin is either

Heads

or

Tails

A spinning coin is temporarily both.

That is similar to a qubit.

4. Entanglement

Entanglement is a phenomenon where two qubits become connected.

Changing one qubit instantly affects the other, even if they are far apart.

Einstein called this

"Spooky action at a distance."

5. Interference

Quantum computers use constructive and destructive interference to amplify correct answers and reduce incorrect ones.

This improves the probability of obtaining the correct solution.

Classical Computer vs Quantum Computer
Feature	Classical Computer	Quantum Computer
Unit	Bit	Qubit
State	0 or 1	0, 1, or both
Processing	Sequential/Parallel	Massive Quantum Parallelism
Speed	Fast	Extremely fast for specific problems
Technology	Silicon Chips	Quantum Physics
Error Rate	Low	High
Stability	Stable	Very fragile
Important Quantum Gates

Quantum gates manipulate qubits.

Hadamard Gate (H)

Creates superposition.

Input

0

Output

50% chance of 0
50% chance of 1
Pauli-X Gate

Equivalent to a NOT gate.

0 → 1

1 → 0
Pauli-Y Gate

Rotates the qubit around the Y-axis.

Pauli-Z Gate

Flips the phase of a qubit.

CNOT Gate

Controlled NOT Gate.

Used to create entanglement.

SWAP Gate

Swaps two qubits.

Toffoli Gate

Three-qubit gate useful for reversible computing.

Quantum Algorithms
Shor's Algorithm

Purpose

Factor very large numbers.

Applications

Breaking RSA encryption
Cryptography research

Complexity

Much faster than classical algorithms.

Grover's Algorithm

Purpose

Search an unsorted database.

Speed

Classical

O(N)

Quantum

O(√N)
Quantum Fourier Transform (QFT)

Used in

Shor's Algorithm
Signal Processing
Quantum Phase Estimation
Quantum Phase Estimation

Used to estimate eigenvalues of quantum systems.

Hardware Technologies

Different companies build quantum computers differently.

Superconducting Qubits

Used by

IBM
Google

Advantages

Fast

Disadvantages

Requires extremely low temperatures.
Trapped Ions

Used by

IonQ

Advantages

Very accurate

Disadvantages

Slower operations
Photonic Quantum Computers

Use photons instead of electrons.

Advantages

Works at room temperature
Neutral Atom Quantum Computers

Use laser-controlled atoms.

Very scalable.

Applications
Drug Discovery

Simulating molecules much faster than classical computers.

Financial Modeling

Portfolio optimization.

Risk analysis.

Fraud detection.

Cryptography

Breaking encryption.

Post-Quantum Cryptography research.

Artificial Intelligence

Faster optimization.

Machine learning acceleration.

Logistics

Delivery route optimization.

Supply chain optimization.

Weather Forecasting

Better climate simulation.

Material Science

Discovering new batteries.

New superconductors.

Better solar cells.

Aerospace

Aircraft design optimization.

Advantages
Extremely fast for certain problems
Can simulate molecules efficiently
Solves optimization problems
Improves AI optimization
Helps scientific research
Better financial modeling
Better logistics optimization
Disadvantages
Very expensive
Extremely fragile
High error rates
Requires cryogenic cooling (for many architectures)
Limited number of qubits
Programming is difficult
Still experimental
Challenges
Decoherence
Quantum noise
Error correction
Scalability
Cost
Hardware complexity
Lack of skilled developers
Real-World Companies
IBM
Google
Microsoft
Amazon
Intel
IonQ
Rigetti
D-Wave
Future Scope

Experts believe quantum computing may revolutionize:

Medicine
AI
Cybersecurity
Climate Science
Finance
Space Exploration
Material Science
National Security
Common Interview Questions
Beginner
What is Quantum Computing?
What is a qubit?
Difference between bit and qubit?
What is superposition?
What is entanglement?
Intermediate
Explain Grover's Algorithm.
Explain Shor's Algorithm.
What is quantum interference?
What are quantum gates?
What is decoherence?
Advanced
Why is error correction difficult in quantum computing?
Compare superconducting and trapped-ion qubits.
Explain Quantum Fourier Transform.
How does quantum computing affect RSA encryption?
What are the limitations of current quantum computers?
Short Summary

Quantum Computing is an emerging computing paradigm based on quantum mechanics. Instead of bits, it uses qubits, which leverage superposition and entanglement to perform computations. While current quantum computers are still in the early stages and face challenges such as error correction and hardware stability, they have the potential to transform fields like cryptography, drug discovery, optimization, artificial intelligence, and scientific research.

Why This Topic Is Great for LangChain Chain Practice

You can build a multi-step chain such as:

User Input
    │
    ▼
Definition Chain
    │
    ▼
Detailed Explanation Chain
    │
    ▼
History Chain
    │
    ▼
Applications Chain
    │
    ▼
Advantages & Disadvantages Chain
    │
    ▼
Interview Questions Chain
    │
    ▼
MCQ Generation Chain
    │
    ▼
    
"""



set_debug(True)
notes_making_model = PromptTemplate(
    template="""
    Creates Notes for the following information
    {textInfo}
    """,
    input_variables=["textInfo"]
)

quiz_maker_mode = PromptTemplate(
    template = """
    Create Quiz type of MCQ from the following Data
    {rawData}
    """,
    input_variables=["rawData"]
)

chain = notes_making_model | llm | StrOutputParser()
NoteData = (chain.invoke(
    {
        "textInfo":textInformation
    }
))
print("These are NOTES  \n ========================================>>>>>> ")
print(NoteData)


chainQuiz = quiz_maker_mode | llm | StrOutputParser()
QuizData = chainQuiz.invoke({
    "rawData" : NoteData
})

print(QuizData)
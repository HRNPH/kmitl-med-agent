# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Experiment Execution
#
# This notebook runs the KMITL Medical Agent experiment with AutoGen agents and MCP integration.

# ## 1. Main Experiment Class


class KMITLMedicalAgent:
    """Main experiment class for KMITL Medical Agent"""

    def __init__(self, test_data, agents, mcp_agent):
        self.test_data = test_data
        self.agents = agents
        self.mcp_agent = mcp_agent
        self.results = []

    async def process_question(
        self, question: str, question_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process a single medical question"""
        try:
            # Start conversation between agents
            user_proxy = self.agents["user_proxy"]
            medical_assistant = self.agents["medical_assistant"]

            # Create the conversation
            chat_history = await user_proxy.a_initiate_chat(
                medical_assistant,
                message=f"""
                Question ID: {question_id if question_id else 'N/A'}
                Question: {question}
                
                Please provide a comprehensive answer to this medical question.
                Consider the Thai healthcare context and provide practical information.
                """,
                max_turns=5,
            )

            # Extract the response
            response = (
                chat_history[-1]["content"] if chat_history else "No response generated"
            )

            return {
                "question_id": question_id,
                "question": question,
                "response": response,
                "status": "success",
            }

        except Exception as e:
            return {
                "question_id": question_id,
                "question": question,
                "response": f"Error processing question: {str(e)}",
                "status": "error",
            }

    async def run_experiment(self, num_questions: int = 5) -> List[Dict[str, Any]]:
        """Run the experiment with test questions"""
        print(f"Starting KMITL Medical Agent Experiment")
        print(f"Processing {num_questions} questions...")

        # Setup MCP connection
        await self.mcp_agent.setup()

        # Process test questions
        questions_to_process = self.test_data.head(num_questions)

        for idx, row in questions_to_process.iterrows():
            question_id = row.get("id", idx + 1)
            question = row.get("question", "")

            print(f"\nProcessing Question {question_id}: {question[:100]}...")

            result = await self.process_question(question, question_id)
            self.results.append(result)

            # Add delay to avoid overwhelming the LLM
            await asyncio.sleep(1)

        # Cleanup MCP connection
        await self.mcp_agent.cleanup()

        return self.results

    def save_results(self, filename: str = "experiment_results.json"):
        """Save experiment results to file"""
        output_path = Path.cwd() / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        print(f"Results saved to: {output_path}")

    def print_results_summary(self):
        """Print a summary of the experiment results"""
        print("\n" + "=" * 50)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 50)

        successful = sum(1 for r in self.results if r["status"] == "success")
        failed = sum(1 for r in self.results if r["status"] == "error")

        print(f"Total Questions Processed: {len(self.results)}")
        print(f"Successful Responses: {successful}")
        print(f"Failed Responses: {failed}")
        print(
            f"Success Rate: {(successful/len(self.results)*100):.1f}%"
            if self.results
            else "0%"
        )

        # Show sample responses
        print("\nSample Responses:")
        for i, result in enumerate(self.results[:3]):  # Show first 3
            print(f"\nQuestion {i+1}: {result['question'][:100]}...")
            print(f"Response: {result['response'][:200]}...")


# ## 2. Initialize Experiment

# Create the main experiment agent
experiment_agent = KMITLMedicalAgent(test_data, agents, mcp_agent)

print("✓ Experiment agent initialized")

# ## 3. Run Experiment


async def run_experiment(num_questions: int = 5):
    """Run the experiment"""
    print(f"\nRunning experiment with {num_questions} questions...")

    # Run the experiment
    results = await experiment_agent.run_experiment(num_questions=num_questions)

    # Save and display results
    experiment_agent.save_results()
    experiment_agent.print_results_summary()

    return results


# ## 4. Execute Experiment

# Uncomment the line below to run the experiment
# results = await run_experiment(num_questions=3)

print("✓ Experiment execution setup completed!")
print("To run the experiment, uncomment the last line in this cell")

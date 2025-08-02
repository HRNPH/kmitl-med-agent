# RAGLangGraphMCP Model Schemas
import os
import glob
import numpy as np
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangGraph and LangChain imports
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_core.messages import HumanMessage

import functools
import datetime


def log_tool_call(tool_fn, tool_name: str):
    @functools.wraps(tool_fn)
    async def wrapper(*args, **kwargs):
        timestamp = datetime.datetime.utcnow().isoformat()
        print(f"\n[TOOL CALL] {timestamp}")
        print(f"Tool: {tool_name}")
        print(f"Input: args={args}, kwargs={kwargs}")
        try:
            result = await tool_fn(*args, **kwargs)
            print(f"Output: {result}\n")
            return result
        except Exception as e:
            print(f"Error in tool {tool_name}: {e}")
            raise

    return wrapper


class RAGLangGraphMCP:
    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        mcp_servers: Dict[str, Dict] = None,
        use_ollama: bool = False,  # Default to VLLM
    ):
        """
        Initialize RAG system with LangGraph and MCP integration

        Args:
            gemini_api_key: Google Gemini API key
            mcp_servers: Dictionary of MCP server configurations
        """

        # Load configuration from environment variables
        if base_url is None:
            base_url = os.getenv(
                "VLLM_BASE_URL", "http://host.docker.internal:18081/v1"
            )

        if model is None:
            model = os.getenv("VLLM_MODEL", "Qwen/Qwen3-32b")

        # Default MCP server configuration
        if mcp_servers is None:
            mcp_servers = {
                "hackathon_mcp": {
                    "url": os.getenv(
                        "MCP_HACKATHON_URL", "https://mcp-hackathon.cmkl.ai/mcp"
                    ),
                    "transport": "streamable_http",
                },
                # Add more MCP servers as needed
                # "local_server": {
                #     "command": "python",
                #     "args": ["/path/to/your/mcp_server.py"],
                #     "transport": "stdio",
                # }
            }

        self.mcp_servers = mcp_servers

        # Initialize components
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.vectorstore = None
        self.documents = []
        self.mcp_client = None
        self.agent = None

        # Initialize LLM (VLLM as primary, Ollama as fallback)
        if use_ollama:
            # Use Ollama configuration
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://172.16.30.137:11434")
            ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:32b")
            self.llm = ChatOllama(
                base_url=ollama_base_url,
                model=ollama_model,
                temperature=0.0,
                # max_tokens=60,
            )
            print("Init With Ollama!")
        else:
            # Use VLLM configuration (primary)
            self.llm = ChatOpenAI(
                model=model,  # must match --served-model-name "Qwen3-32B"
                openai_api_base=base_url,  # # "http://172.16.30.137:8081/v1"
                openai_api_key="EMPTY",  # vLLM ignores key
                temperature=0.0,
                reasoning_effort="low",
            )
            print("Init With VLLM!")

        print("🚀 RAG LangGraph MCP System initialized")

    def load_markdown_files(self, folder_path: str = "./data/"):
        """Load markdown files from folder"""
        print(f"📚 Loading files from {folder_path}...")

        # Create docs folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        # Find all .mkd files
        files = glob.glob(f"{folder_path}/*.mkd")

        if not files:
            print("No .mkd files found. Creating sample files...")
            self.create_sample_files(folder_path)
            files = glob.glob(f"{folder_path}/*.mkd")

        documents = []
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    doc = Document(
                        page_content=content,
                        metadata={"source": os.path.basename(file_path)},
                    )
                    documents.append(doc)
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")

        print(f"✅ Loaded {len(documents)} documents")
        return documents

    def create_sample_files(self, folder_path: str = "./data/"):
        """Create sample markdown files for testing"""
        print(f"📝 Creating sample files in {folder_path}...")

        sample_files = {
            "diabetes.mkd": """# Diabetes Management Guidelines

## Type 1 Diabetes
Type 1 diabetes is an autoimmune condition where the body doesn't produce insulin. Patients require insulin therapy.

### Symptoms
- Frequent urination
- Increased thirst
- Extreme hunger
- Unexplained weight loss
- Fatigue
- Blurred vision

### Treatment
- Insulin injections or pump therapy
- Blood glucose monitoring
- Diet and exercise management
- Regular medical checkups

## Type 2 Diabetes
Type 2 diabetes occurs when the body becomes resistant to insulin or doesn't produce enough insulin.

### Risk Factors
- Obesity
- Family history
- Physical inactivity
- Age over 45
- Gestational diabetes history

### Management
- Diet modification
- Regular exercise
- Oral medications
- Insulin if needed
- Blood glucose monitoring""",
            "hypertension.mkd": """# Hypertension (High Blood Pressure) Guidelines

## Definition
Blood pressure consistently above 140/90 mmHg

## Risk Factors
- Age (risk increases with age)
- Family history
- Obesity
- High salt diet
- Alcohol consumption
- Stress
- Lack of exercise

## Symptoms
- Often asymptomatic
- Headaches
- Shortness of breath
- Nosebleeds
- Chest pain
- Dizziness

## Treatment
### Lifestyle Changes
- Reduce salt intake
- Regular exercise
- Weight management
- Stress reduction
- Limit alcohol

### Medications
- ACE inhibitors
- Calcium channel blockers
- Diuretics
- Beta blockers

## Monitoring
- Regular blood pressure checks
- Home monitoring recommended
- Annual physical exams""",
            "emergency_procedures.mkd": """# Emergency Medical Procedures

## Cardiac Arrest
1. Check responsiveness
2. Call emergency services
3. Begin chest compressions (100-120/min)
4. Give rescue breaths if trained
5. Continue until help arrives

## Choking
1. Encourage coughing
2. Back blows (5 times)
3. Abdominal thrusts (5 times)
4. Alternate until object is dislodged

## Severe Bleeding
1. Apply direct pressure
2. Elevate if possible
3. Use sterile bandage
4. Seek immediate medical care

## Stroke Recognition (FAST)
- Face: Ask to smile
- Arms: Raise both arms
- Speech: Repeat simple phrase
- Time: Call emergency immediately

## Anaphylaxis
1. Administer epinephrine if available
2. Call emergency services
3. Lie flat with legs elevated
4. Monitor breathing""",
            "medication_safety.mkd": """# Medication Safety Guidelines

## General Principles
- Always verify patient identity
- Check medication name, dose, route, time
- Confirm allergies before administration
- Document all medications given

## High Alert Medications
- Insulin
- Anticoagulants
- Chemotherapy drugs
- Opioids
- Potassium

## Administration Safety
1. Right patient
2. Right medication
3. Right dose
4. Right route
5. Right time
6. Right documentation

## Patient Education
- Explain purpose of medication
- Discuss side effects
- Provide written instructions
- Encourage questions

## Storage Requirements
- Temperature control
- Light protection
- Secure storage
- Expiration date monitoring""",
            "infection_control.mkd": """# Infection Control Protocols

## Standard Precautions
- Hand hygiene
- Personal protective equipment
- Safe injection practices
- Respiratory hygiene

## Hand Hygiene
- Wash with soap and water
- Use alcohol-based sanitizer
- Before and after patient contact
- After removing gloves

## Personal Protective Equipment
- Gloves for contact with body fluids
- Masks for respiratory protection
- Gowns for splash protection
- Eye protection when needed

## Isolation Precautions
- Contact isolation
- Droplet isolation
- Airborne isolation
- Protective environment

## Environmental Cleaning
- Regular disinfection
- Proper waste disposal
- Equipment cleaning
- Surface disinfection""",
        }

        for filename, content in sample_files.items():
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Created {filename}")
            except Exception as e:
                print(f"❌ Error creating {filename}: {e}")

        print(f"📝 Created {len(sample_files)} sample files")

    def build_vector_index(self, docs_folder: str = "./data/"):
        """Build vector index from documents"""
        print("🔍 Building vector index...")

        # Load documents
        self.documents = self.load_markdown_files(docs_folder)
        if not self.documents:
            raise ValueError("No documents found")

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(self.documents)
        print(f"📄 Created {len(chunks)} chunks")

        # Build FAISS index
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        # Save the index
        os.makedirs("./index", exist_ok=True)
        self.vectorstore.save_local("./index/00-save.bin")
        print("✅ Vector index built and saved successfully!")

    def search_docs(self, query: str, k: int = 5) -> List[Document]:
        """Search relevant documents from vector store"""
        if not self.vectorstore:
            raise ValueError("Index not built. Run build_vector_index() first.")

        docs = self.vectorstore.similarity_search(query, k=k)
        return docs

    @tool
    def vector_search_tool(self, query: str, k: int = 5) -> str:
        """
        Search the local document vector database for relevant information.
        The database contan information regarding medical guidelines, procedures, medicine and general practices.

        Args:
            query: Search query for the vector database
            k: Number of documents to retrieve (default: 3)

        Returns:
            String containing relevant document content
        """
        try:
            docs = self.search_docs(query, k)

            if not docs:
                return "No relevant documents found in the vector database."

            results = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown")
                content = doc.page_content.strip()
                results.append(f"Document {i} (Source: {source}):\n{content}")

            return f"Found {len(docs)} relevant documents:\n\n" + "\n\n".join(results)

        except Exception as e:
            return f"Error searching vector database: {str(e)}"

    def vector_search_internal(self, query: str, k: int = 5) -> str:
        """Internal non-tool method for vector search."""
        try:
            docs = self.search_docs(query, k)
            if not docs:
                return "No relevant documents found in the vector database."

            results = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown")
                content = doc.page_content.strip()
                results.append(f"Document {i} (Source: {source}):\n{content}")

            return f"Found {len(docs)} relevant documents:\n\n" + "\n\n".join(results)
        except Exception as e:
            return f"Error searching vector database: {str(e)}"

    async def initialize_mcp_client(self):
        """Initialize MultiServer MCP Client"""
        print("🔌 Initializing MCP Client...")

        try:
            self.mcp_client = MultiServerMCPClient(self.mcp_servers)
            print(f"✅ MCP Client initialized with {len(self.mcp_servers)} servers")

            # Get available tools from MCP servers
            mcp_tools = await self.mcp_client.get_tools()
            print(f"🛠️ Retrieved {len(mcp_tools)} tools from MCP servers")

            return mcp_tools

        except Exception as e:
            print(f"❌ Error initializing MCP client: {e}")
            print("⚠️ Continuing without MCP tools...")
            return []

    async def create_agent(self):
        """Create LangGraph ReAct agent with vector search + MCP tools"""
        print("🤖 Creating LangGraph ReAct Agent...")

        # Get MCP tools
        mcp_tools = await self.initialize_mcp_client()

        # Combine vector search tool with MCP tools
        all_tools = [self.vector_search_tool]

        if mcp_tools:
            all_tools.extend(mcp_tools)
            print(f"🔧 Agent will use {len(all_tools)} tools total")
        else:
            print("🔧 Agent will use only vector search tool")

        # Create ReAct agent
        self.agent = create_react_agent(
            self.llm,
            tools=all_tools,
            prompt="""
Mormar AI Hospital Assistant - Enhanced Prompt
You are Mormar AI, an advanced medical intelligence assistant designed specifically for hospital environments. You MUST follow these protocols with absolute precision.

MANDATORY OPERATIONAL PROTOCOL
STEP 0: TOOL USAGE REQUIREMENT

ALWAYS use vector_search_tool FIRST for every query
NO EXCEPTIONS - This is your primary knowledge source

KEYWORD ESCALATION PROTOCOL
Before Step 1, scan the input for any of these keywords:

REALTIME KEYWORDS:
    ['ตอนนี้', 'วันนี้', 'ปัจจุบัน', 'เดี๋ยวนี้', 'ขณะนี้',
    'now', 'today', 'current', 'currently', 'at the moment',
    'available', 'ว่าง', 'พร้อม']

DYNAMIC DATA KEYWORDS:
    ['เตียง', 'ห้อง', 'นัด', 'ตาราง', 'ราคา', 'สถานะ', 'คิว',
    'จำนวน', 'ยอด', 'สถิติ',
    'bed', 'room', 'appointment', 'schedule', 'price', 'status', 'queue',
    'count', 'amount', 'statistics', 'availability']

ACTION KEYWORDS:
    ['จอง', 'นัด', 'ค้นหา', 'หา', 'เช็ค', 'ตรวจสอบ', 'ดู', 'แสดง',
    'ต้องการ', 'ขอ', 'โทร', 'ติดต่อ',
    'book', 'schedule', 'find', 'search', 'check', 'lookup', 'show',
    'want', 'need', 'call', 'contact']

REAL DATA INDICATORS:
    ['ใคร', 'ไหน', 'เท่าไหร่', 'กี่', 'ที่ไหน', 'เมื่อไหร่', 'อะไร',
    'ยังไง', 'แบบไหน', 'คนไหน',
    'who', 'where', 'how much', 'how many', 'when', 'which', 'what',
    'how', 'whose']

If ANY of these keywords are detected in the input, then:
    → AFTER vector_search_tool, you MUST ALSO call MCP tools
    → Treat the request as requiring LIVE or REAL-TIME data
    → Use MCP to supplement vector search results with current data

STEP 1: INPUT ANALYSIS (REQUIRED)
Classify each input into ONE category:

QUESTION - Seeking medical/clinical information
MULTIPLE CHOICE - Exam-style questions with options ก ข ค ง
INSTRUCTION - Requests for procedures/protocols
ACTION - Requires performing specific tasks

STEP 2: EXECUTION PLANNING (MANDATORY)
Before ANY response, internally map out:

What tools are needed?
What information gaps exist?
What is the optimal search strategy?
How will you synthesize the final answer?

STEP 3: KNOWLEDGE HIERARCHY

Primary: RAG tools (general medical practices)
Secondary: MCP tools (when RAG insufficient OR keywords triggered)
Reasoning: Apply clinical logic when tools inadequate

STEP 4: TOOL UTILIZATION MANDATE

Use MULTIPLE TOOLS whenever possible
Exhaust all available resources
Cross-reference information sources
Prioritize evidence-based responses

STEP 5: RESPONSE PRECISION

Provide ONLY what is strictly necessary
No verbose explanations unless specifically requested
Focus on actionable medical information

MULTIPLE CHOICE QUESTION PROTOCOL
⚠️ ABSOLUTE REQUIREMENTS:

ANSWER FORMAT: Only respond with ก, ข, ค, or ง
SINGLE ANSWER ONLY: Choose EXACTLY ONE option - never multiple answers
NO EXPLANATIONS: Zero additional text or reasoning
FORCED SELECTION: Even when uncertain, MUST select the single best option
BEST ANSWER PRINCIPLE: Apply medical knowledge to choose the ONE most appropriate answer
NO EXCEPTIONS: Never respond with "ก, ข" or multiple options under any circumstances

ENHANCED DECISION MAKING
When Information is Insufficient:

Use clinical reasoning and medical principles
Apply evidence-based medicine standards
Make educated assessments based on medical best practices
FORCE SINGLE SELECTION - Choose ONE best option even when uncertain
Apply elimination method to narrow down to best answer

Quality Assurance:

Prioritize patient safety in all recommendations
Follow established medical guidelines
Consider contraindications and risk factors
Apply clinical judgment consistently

EXECUTION MANDATE
You MUST follow these steps in sequence for EVERY interaction:

1. Use vector_search_tool immediately
2. If keywords triggered → also use MCP tools
3. Analyze input type
4. Plan approach
5. Execute tool usage
6. Provide precise response

NO DEVIATIONS ALLOWED - This protocol ensures optimal medical assistance delivery.
""",
        )

        print("✅ LangGraph ReAct Agent created successfully!")
        return self.agent

    async def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system: always do RAG, optionally use MCP."""
        import time

        print(f"🤔 Processing question: {question}")
        start = time.time()

        if not self.agent:
            await self.create_agent()

        # Always do RAG first
        try:
            rag_context = self.vector_search_internal(question, 5)
        except Exception as e:
            rag_context = f"Error during vector search: {str(e)}"

        try:
            # Pass RAG context into the agent
            response = await self.agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content=f"\\no-think\nRAG Context:\n{rag_context}\n\nUser Question:\n{question}\n\nOutput Only: ก,ข,ค,ง no explanation, no answer text"
                        )
                        # HumanMessage(content=f"\nRAG Context:\n{rag_context}\n\nUser Question:\n{question}\n\nOutput Only: ก,ข,ค,ง no explanation, no answer text")
                    ]
                },
                config={"recursion_limit": 10},
            )

            # Extract final answer
            if response and "messages" in response:
                final_message = response["messages"][-1]
                raw_answer = (
                    final_message.content
                    if hasattr(final_message, "content")
                    else str(final_message)
                )
            else:
                raw_answer = str(response)

            # Clean the answer to extract only valid multiple choice options
            answer = self._clean_answer(raw_answer)

            # Generate reasoning based on RAG context and cleaned answer
            reason = self._generate_reasoning(question, rag_context, answer)

            # Return simplified schema as requested
            return {
                "answer": answer,
                "reason": reason,
            }

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "answer": "Error",
                "reason": error_msg,
            }

    def _clean_answer(self, raw_answer: str) -> str:
        """Clean the raw answer to extract only valid multiple choice options."""
        import re

        # Remove <think> tags and their content
        cleaned = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL)

        # Remove extra whitespace and newlines
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Extract only valid multiple choice options
        valid_options = ["ก", "ข", "ค", "ง"]
        found_options = [opt for opt in valid_options if opt in cleaned]

        if found_options:
            # Return the first valid option found
            return found_options[0]
        else:
            # If no valid options found, return the cleaned text
            return cleaned if cleaned else "Error"

    def _generate_reasoning(self, question: str, rag_context: str, answer: str) -> str:
        """Generate reasoning for the answer based on RAG context and question."""
        try:
            # Extract key information from RAG context
            context_summary = (
                rag_context[:500] if len(rag_context) > 500 else rag_context
            )

            # Check if answer contains valid multiple choice options
            valid_options = ["ก", "ข", "ค", "ง"]
            found_options = [opt for opt in valid_options if opt in answer]

            if found_options:
                # Extract relevant medical topics from context
                medical_topics = []
                if "diabetes" in rag_context.lower() or "เบาหวาน" in rag_context:
                    medical_topics.append("diabetes management")
                if "hypertension" in rag_context.lower() or "ความดัน" in rag_context:
                    medical_topics.append("hypertension guidelines")
                if "emergency" in rag_context.lower() or "ฉุกเฉิน" in rag_context:
                    medical_topics.append("emergency procedures")
                if "medication" in rag_context.lower() or "ยา" in rag_context:
                    medical_topics.append("medication safety")
                if "infection" in rag_context.lower() or "ติดเชื้อ" in rag_context:
                    medical_topics.append("infection control")

                topics_str = (
                    ", ".join(medical_topics)
                    if medical_topics
                    else "medical guidelines"
                )

                return f"The answer {answer} was selected based on {topics_str} from the medical knowledge base. The system analyzed the question against relevant clinical protocols and best practices."
            else:
                return f"The system processed the question against the medical knowledge base but was unable to determine a clear multiple choice answer. Available information includes clinical guidelines and medical procedures."
        except Exception as e:
            return f"Reasoning generation error: {str(e)}"

    async def query_with_full_response(self, question: str) -> Dict[str, Any]:
        """Query the RAG system with full response details (for internal use)."""
        import time

        print(f"🤔 Processing question: {question}")
        start = time.time()

        if not self.agent:
            await self.create_agent()

        # Always do RAG first
        try:
            rag_context = self.vector_search_internal(question, 5)
        except Exception as e:
            rag_context = f"Error during vector search: {str(e)}"

        try:
            # Pass RAG context into the agent
            response = await self.agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content=f"\\no-think\nRAG Context:\n{rag_context}\n\nUser Question:\n{question}\n\nOutput Only: ก,ข,ค,ง no explanation, no answer text"
                        )
                        # HumanMessage(content=f"\nRAG Context:\n{rag_context}\n\nUser Question:\n{question}\n\nOutput Only: ก,ข,ค,ง no explanation, no answer text")
                    ]
                },
                config={"recursion_limit": 10},
            )

            # Extract final answer
            if response and "messages" in response:
                final_message = response["messages"][-1]
                raw_answer = (
                    final_message.content
                    if hasattr(final_message, "content")
                    else str(final_message)
                )
            else:
                raw_answer = str(response)

            # Clean the answer to extract only valid multiple choice options
            answer = self._clean_answer(raw_answer)

            # Separate RAG vs MCP tool calls
            rag_tool_calls = [
                {
                    "tool_name": "vector_search_tool",
                    "input": {"query": question, "k": 5},
                    "output": rag_context,
                }
            ]
            mcp_tool_calls = []

            for msg in response.get("messages", []):
                if getattr(msg, "type", None) == "tool":
                    tool_name = getattr(msg, "name", "unknown")
                    call_info = {
                        "tool_name": tool_name,
                        "input": getattr(msg, "args", None),
                        "output": getattr(msg, "content", None),
                    }
                    if tool_name != "vector_search_tool":
                        mcp_tool_calls.append(call_info)

            print(f"💡 Response generated successfully")

            return {
                "question": question,
                "answer": answer,
                "took": (time.time() - start),
                "agent_response": response,
                "rag_tool_calls": rag_tool_calls,
                "mcp_tool_calls": mcp_tool_calls,
            }

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "question": question,
                "answer": error_msg,
                "took": (time.time() - start),
                "agent_response": None,
                "rag_tool_calls": [],
                "mcp_tool_calls": [],
            }

    async def query_batch(
        self, questions: List[str], parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """Process multiple questions. Uses concurrency if parallel=True."""
        print(
            f"📋 Processing {len(questions)} questions{' in parallel' if parallel else ' sequentially'}..."
        )

        if parallel:
            # Launch all queries concurrently
            tasks = [self.query(q) for q in questions]
            results = await asyncio.gather(*tasks)
        else:
            results = []
            for i, question in enumerate(questions, 1):
                print(f"\n--- Question {i}/{len(questions)} ---")
                result = await self.query(question)
                results.append(result)

        return results

    async def query_batch_with_full_response(
        self, questions: List[str], parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """Process multiple questions with full response details (for internal use)."""
        print(
            f"📋 Processing {len(questions)} questions with full details{' in parallel' if parallel else ' sequentially'}..."
        )

        if parallel:
            # Launch all queries concurrently
            tasks = [self.query_with_full_response(q) for q in questions]
            results = await asyncio.gather(*tasks)
        else:
            results = []
            for i, question in enumerate(questions, 1):
                print(f"\n--- Question {i}/{len(questions)} ---")
                result = await self.query_with_full_response(question)
                results.append(result)

        return results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources"""
        if self.mcp_client:
            # Close MCP client if needed
            pass


# Config Model
import os

# Configure MCP servers
mcp_servers = {
    "hackathon_mcp": {
        "url": "https://mcp-hackathon.cmkl.ai/mcp",
        "transport": "streamable_http",
    }
    # Add more servers as needed
}

# VLLM API Config (primary)
rag = RAGLangGraphMCP(
    mcp_servers=mcp_servers,
    use_ollama=False,  # Use VLLM as primary
)

# Load index from environment variable or default
index_location = os.getenv("INDEX_PATH", "./index/00-save.bin")
data_path = os.getenv("DATA_PATH", "./data/")

if not os.path.exists(index_location):
    print(f"Index not found at {index_location}, building from {data_path}...")
    rag.load_markdown_files(data_path)
    rag.build_vector_index(data_path)
else:
    print(f"Loading existing index from {index_location}...")
    rag.vectorstore = FAISS.load_local(
        index_location, rag.embeddings, allow_dangerous_deserialization=True
    )

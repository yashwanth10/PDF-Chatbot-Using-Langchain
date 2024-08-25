**Overview** <br/>
The PDF Document Chatbot is a Streamlit-based application that allows users to upload PDF files and interact with them using natural language queries. The chatbot leverages the LangChain and HuggingFace libraries to process and embed text from the uploaded PDFs, and uses the GROQ API of Llama-3.1-70b-Versatile for generating responses.

**Features**<br/>
Upload PDF Files: Users can upload multiple PDF files.
Chat with Documents: Users can ask questions about the content of the uploaded PDFs.
Document Embedding: The application creates vector embeddings of the PDF content to facilitate efficient querying.
Chat History: Displays a history of user queries and chatbot responses.
Document Similarity Search: Shows relevant parts of the documents related to the query.
Requirements
To run this application, you'll need the following Python packages:

**Requirements**<br/>
streamlit <br/>
langchain_groq<br/>
langchain_community<br/>
langchain_text_splitters<br/>
langchain_core<br/>
langchain_huggingface<br/>
dotenv<br/>
faiss-cpu.

All the necessary packages have been added in the requirements.txt file and the python environment used is of version 3.10. 
Run the following command to install all the packages :

pip install -r requirements.txt

**Usage**<br/>
Enter API Keys:
When you first open the app, you will be prompted to enter your GROQ API Key.

Upload PDFs:
Use the file uploader to upload one or multiple PDF files.

Chat with Documents:
Enter your query in the text area and click "Let's chat" to get responses based on the content of the uploaded PDFs.

View Chat History:
The chat history is displayed with the most recent responses at the top.

Document Similarity Search:
Expand the "Document similarity search" section to view parts of the documents relevant to your query.

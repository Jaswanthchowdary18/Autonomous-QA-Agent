"""
Intelligent QA Agent - Streamlit Frontend
Dynamically interacts with the intelligent backend
"""

import streamlit as st
import requests
import json
import os
import tempfile
from typing import List, Dict, Any
import time

# Configuration
BACKEND_URL = "http://localhost:8000"

def init_session_state():
    """Initialize session state variables"""
    if 'documents_analyzed' not in st.session_state:
        st.session_state.documents_analyzed = False
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'test_cases' not in st.session_state:
        st.session_state.test_cases = []
    if 'detected_features' not in st.session_state:
        st.session_state.detected_features = []
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'selected_test_case' not in st.session_state:
        st.session_state.selected_test_case = None
    if 'generated_script' not in st.session_state:
        st.session_state.generated_script = ""
    if 'script_metadata' not in st.session_state:
        st.session_state.script_metadata = {}

def check_backend_connection():
    """Check if backend is accessible"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def call_backend(endpoint: str, method: str = "GET", data: dict = None, files: dict = None):
    """Make API call to backend with proper error handling"""
    try:
        url = f"{BACKEND_URL}/{endpoint.lstrip('/')}"
        
        if method == "GET":
            response = requests.get(url, params=data, timeout=30)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, timeout=30)
            else:
                response = requests.post(url, json=data, timeout=30)
        else:
            return {"status": "error", "message": f"Unsupported method: {method}"}
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "message": f"Backend returned status {response.status_code}: {response.text}"}
            
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Cannot connect to backend. Make sure the FastAPI server is running on port 8000."}
    except Exception as e:
        return {"status": "error", "message": f"API call failed: {str(e)}"}

def main():
    st.set_page_config(
        page_title="Intelligent QA Agent",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🧠 Intelligent QA Agent")
    st.markdown("**Dynamically** generates test cases based on your project documentation")
    
    init_session_state()
    
    # Check backend connection
    if not check_backend_connection():
        st.error("""
        ❌ **Cannot connect to backend**
        
        Please ensure:
        1. **Backend server is running** - You should see FastAPI running on http://0.0.0.0:8000
        2. **Run this command:** `python app.py` in the backend directory
        3. **Wait for the message:** "Uvicorn running on http://0.0.0.0:8000"
        """)
        return
    
    st.sidebar.success("✅ Backend Connected")
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "📁 Document Analysis", 
        "🧪 Test Case Generation", 
        "💻 Script Generation",
        "🔍 Feature Analysis"
    ])
    
    if page == "📁 Document Analysis":
        show_document_analysis()
    elif page == "🧪 Test Case Generation":
        show_test_generation()
    elif page == "💻 Script Generation":
        show_script_generation()
    elif page == "🔍 Feature Analysis":
        show_feature_analysis()

def show_document_analysis():
    st.header("📁 Document Analysis & Feature Extraction")
    st.info("Upload your project documents to automatically extract features and requirements")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Project Documents")
        
        uploaded_files = st.file_uploader(
            "Choose project documentation files",
            type=['md', 'txt', 'json', 'html', 'pdf'],
            accept_multiple_files=True,
            help="Upload product specifications, UI/UX guides, API documentation, HTML files, etc."
        )
        
        if uploaded_files:
            st.success(f"📄 {len(uploaded_files)} files ready for analysis")
            
            with st.expander("📋 Uploaded Files Preview", expanded=True):
                for file in uploaded_files:
                    file_size = len(file.getvalue()) / 1024  # KB
                    st.write(f"• **{file.name}** ({file_size:.1f} KB)")
                    
                    # Show small preview for text files
                    if file.name.endswith(('.md', '.txt', '.html')):
                        try:
                            content = file.getvalue().decode('utf-8')
                            preview = content[:200] + "..." if len(content) > 200 else content
                            st.code(preview, language='text' if file.name.endswith('.txt') else 'html')
                        except:
                            st.info("Binary file - cannot preview")
    
    with col2:
        st.subheader("Analysis Status")
        
        if st.session_state.documents_analyzed:
            st.success("✅ Documents Analyzed")
            st.write(f"**Detected Features:** {len(st.session_state.detected_features)}")
            st.write(f"**Test Cases Ready:** {len(st.session_state.test_cases)}")
        else:
            st.warning("⏳ Waiting for document analysis")
    
    # Analyze Documents Button
    st.markdown("---")
    st.subheader("Extract Features from Documents")
    
    if st.button("🔍 Analyze Documents & Extract Features", type="primary", use_container_width=True):
        if not uploaded_files:
            st.error("Please upload at least one document first")
            return
        
        with st.spinner("🔬 Analyzing documents and extracting features..."):
            # Prepare files for upload
            files_to_upload = []
            for uploaded_file in uploaded_files:
                files_to_upload.append(('files', (uploaded_file.name, uploaded_file.getvalue(), 'application/octet-stream')))
            
            # Call backend to analyze documents
            result = call_backend("upload-documents", "POST", files=files_to_upload)
            
            if result.get("status") == "success":
                st.session_state.documents_analyzed = True
                st.session_state.uploaded_files = uploaded_files
                st.session_state.detected_features = result.get("detected_features", [])
                st.session_state.analysis_results = result.get("analyzed_documents", [])
                
                st.success(f"✅ {result['message']}")
                st.balloons()
                
                # Show summary
                summary = result.get("summary", {})
                st.subheader("📊 Analysis Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Files Processed", f"{summary.get('successful_files', 0)}/{summary.get('total_files', 0)}")
                with col2:
                    st.metric("Features Detected", summary.get('total_features_detected', 0))
                with col3:
                    st.metric("Workflows Found", summary.get('total_workflows', 0))
                with col4:
                    st.metric("UI Elements", summary.get('total_ui_elements', 0))
                
                # Show feature detection results
                if st.session_state.detected_features:
                    st.subheader("🎯 Detected Features")
                    features_grid = st.columns(3)
                    for i, feature in enumerate(st.session_state.detected_features):
                        with features_grid[i % 3]:
                            st.info(f"**{feature.replace('_', ' ').title()}**")
                
                # Show detailed analysis
                with st.expander("📋 Detailed Analysis Results"):
                    for file_result in result.get("processed_files", []):
                        if file_result.get("status") == "success":
                            st.write(f"**📄 {file_result['filename']}**")
                            analysis = file_result.get("analysis", {})
                            st.write(f"Features: {analysis.get('features_found', 0)} | "
                                   f"Workflows: {analysis.get('workflows_identified', 0)} | "
                                   f"UI Elements: {analysis.get('ui_elements_detected', 0)}")
                        else:
                            st.error(f"❌ {file_result['filename']}: {file_result.get('error', 'Unknown error')}")
                            
            else:
                st.error(f"❌ {result.get('message', 'Analysis failed')}")

def show_test_generation():
    st.header("🧪 Intelligent Test Case Generation")
    
    if not st.session_state.documents_analyzed:
        st.warning("""
        ⚠️ **Please analyze documents first**
        
        Go to **Document Analysis** page and upload your project documents to enable test case generation.
        """)
        return
    
    st.success(f"🎯 **Detected Features:** {', '.join(st.session_state.detected_features)}")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Generate Test Cases")
        
        # Smart query suggestions based on detected features
        feature_suggestions = {
            "login": "Generate test cases for login functionality",
            "checkout": "Test checkout process and payment validation",
            "search": "Create search functionality test cases", 
            "form": "Test form validation and user input scenarios",
            "cart": "Generate shopping cart functionality tests",
            "payment": "Create payment method test cases"
        }
        
        suggested_queries = []
        for feature in st.session_state.detected_features:
            suggestion = feature_suggestions.get(feature, f"Test {feature} functionality")
            suggested_queries.append(suggestion)
        
        default_query = suggested_queries[0] if suggested_queries else "Generate comprehensive test cases"
        
        query = st.text_area(
            "What would you like to test?",
            value=default_query,
            height=80,
            help="Be specific about the features you want to test. The AI will generate relevant test cases based on your documents."
        )
        
        # Query suggestions
        if suggested_queries:
            st.write("💡 **Suggested queries:**")
            for suggestion in suggested_queries[:3]:
                if st.button(suggestion, key=f"suggest_{suggestion}"):
                    st.session_state.current_query = suggestion
                    st.rerun()
    
    with col2:
        st.subheader("Actions")
        if st.button("🚀 Generate Test Cases", type="primary", use_container_width=True):
            with st.spinner("🤖 Generating intelligent test cases..."):
                result = call_backend("generate-test-cases", "POST", {"query": query})
                
                if result.get("status") == "success":
                    st.session_state.test_cases = result.get("test_cases", [])
                    st.session_state.generation_query = query
                    st.session_state.generation_id = result.get("generation_id")
                    
                    st.success(f"✅ Generated {len(st.session_state.test_cases)} test cases")
                else:
                    st.error(f"❌ {result.get('message', 'Test generation failed')}")
        
        if st.button("🔄 Clear Results", use_container_width=True):
            st.session_state.test_cases = []
            st.rerun()
    
    # Display generated test cases
    if st.session_state.test_cases:
        st.markdown("---")
        st.subheader(f"📋 Generated Test Cases ({len(st.session_state.test_cases)} total)")
        
        # Test case statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            positive_tests = len([tc for tc in st.session_state.test_cases if tc.get('test_type') == 'positive'])
            st.metric("Positive Tests", positive_tests)
        with col2:
            negative_tests = len([tc for tc in st.session_state.test_cases if tc.get('test_type') == 'negative'])
            st.metric("Negative Tests", negative_tests)
        with col3:
            features_covered = len(set(tc.get('feature', 'unknown') for tc in st.session_state.test_cases))
            st.metric("Features Covered", features_covered)
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            all_features = list(set(tc.get("feature", "unknown") for tc in st.session_state.test_cases))
            feature_filter = st.selectbox("Filter by Feature", ["All Features"] + all_features)
        with col2:
            test_type_filter = st.selectbox("Filter by Test Type", ["All Types", "positive", "negative", "boundary"])
        
        # Filter test cases
        filtered_cases = st.session_state.test_cases
        if feature_filter != "All Features":
            filtered_cases = [tc for tc in filtered_cases if tc.get("feature") == feature_filter]
        if test_type_filter != "All Types":
            filtered_cases = [tc for tc in filtered_cases if tc.get("test_type") == test_type_filter]
        
        st.info(f"Showing {len(filtered_cases)} test cases")
        
        # Display test cases
        for i, test_case in enumerate(filtered_cases):
            with st.expander(f"🔬 Test Case {i+1}: {test_case.get('name', 'Unnamed Test')}", expanded=i < 2):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Test case details
                    st.write(f"**Feature:** {test_case.get('feature', 'Unknown')}")
                    st.write(f"**Test Type:** {test_case.get('test_type', 'general')}")
                    st.write(f"**Priority:** {test_case.get('priority', 'Medium')}")
                    st.write(f"**Description:** {test_case.get('description', 'No description')}")
                    
                    # Test steps
                    if 'steps' in test_case:
                        st.write("**Test Steps:**")
                        for j, step in enumerate(test_case['steps'], 1):
                            st.write(f"{j}. {step}")
                    
                    # Expected result
                    st.write(f"**Expected Result:** {test_case.get('expected_result', 'Success')}")
                    
                    # Confidence score
                    confidence = test_case.get('confidence_score', 0)
                    st.write(f"**Confidence Score:** {confidence:.2f}")
                
                with col2:
                    # Selection for script generation
                    if st.button(f"📝 Generate Script", key=f"script_{i}", use_container_width=True):
                        st.session_state.selected_test_case = test_case
                        st.success(f"Selected for script generation!")
                        # Switch to script generation page
                        st.rerun()

def show_script_generation():
    st.header("💻 Selenium Script Generation")
    
    if not st.session_state.documents_analyzed:
        st.warning("⚠️ Please analyze documents first in the Document Analysis section")
        return
    
    if 'selected_test_case' not in st.session_state or not st.session_state.selected_test_case:
        st.info("""
        ℹ️ **No test case selected**
        
        Please go to **Test Case Generation** and select a test case to generate its Selenium script.
        """)
        
        # Show quick selection from available test cases
        if st.session_state.test_cases:
            st.subheader("Quick Selection")
            for i, test_case in enumerate(st.session_state.test_cases[:5]):  # Show first 5
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{test_case.get('name', 'Unnamed Test')}**")
                    st.write(f"Feature: {test_case.get('feature', 'Unknown')}")
                with col2:
                    if st.button(f"Select", key=f"quick_{i}"):
                        st.session_state.selected_test_case = test_case
                        st.rerun()
        return
    
    # Display selected test case
    selected_case = st.session_state.selected_test_case
    st.success(f"✅ **Selected:** {selected_case.get('name', 'Unnamed Test')}")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("⚡ Generate Selenium Script", type="primary", use_container_width=True):
            with st.spinner("💻 Generating executable Selenium script..."):
                result = call_backend("generate-script", "POST", {
                    "test_case": selected_case,
                    "test_case_id": selected_case.get('id', 'unknown')
                })
                
                if result.get("status") == "success":
                    # FIX: Handle both string and dictionary response formats
                    script_data = result.get("script", "")
                    if isinstance(script_data, dict):
                        # If script is a dictionary, extract the actual script content
                        st.session_state.generated_script = script_data.get('code', '') or script_data.get('content', '') or str(script_data)
                    else:
                        st.session_state.generated_script = script_data
                    
                    st.session_state.script_metadata = result
                    st.success("✅ Selenium script generated successfully!")
                else:
                    st.error(f"❌ {result.get('message', 'Script generation failed')}")
    
    with col2:
        if st.button("🔄 Change Test Case", use_container_width=True):
            st.session_state.selected_test_case = None
            if 'generated_script' in st.session_state:
                del st.session_state.generated_script
            st.rerun()
    
    # Display generated script
    if 'generated_script' in st.session_state:
        st.markdown("---")
        st.subheader("📜 Generated Selenium Script")
        
        # Script information
        with st.expander("📊 Script Information", expanded=True):
            st.write(f"**Test Case:** {selected_case.get('name', 'Unnamed Test')}")
            st.write(f"**Feature:** {selected_case.get('feature', 'Unknown')}")
            st.write(f"**Test Type:** {selected_case.get('test_type', 'general')}")
            
            if 'steps' in selected_case:
                st.write("**Test Steps:**")
                for i, step in enumerate(selected_case['steps'], 1):
                    st.write(f"{i}. {step}")
            
            # Show script metadata if available
            if st.session_state.script_metadata:
                metadata = st.session_state.script_metadata.get('metadata', {})
                if metadata:
                    st.write("**Script Metadata:**")
                    st.write(f"- Has Evidence: {metadata.get('has_evidence', 'No')}")
                    st.write(f"- Feature: {metadata.get('feature', 'Unknown')}")
                    st.write(f"- Test Type: {metadata.get('test_type', 'general')}")
        
        # The script itself - ensure it's a string
        script_content = st.session_state.generated_script
        if isinstance(script_content, dict):
            # Extract script content from dictionary
            script_content = script_content.get('code', '') or script_content.get('content', '') or json.dumps(script_content, indent=2)
        
        st.code(script_content, language='python')
        
        # Download section - FIXED: Handle dictionary properly
        st.subheader("📥 Download Script")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # FIX: Ensure we're working with string data for download
            if isinstance(script_content, str):
                script_bytes = script_content.encode('utf-8')
            else:
                script_bytes = str(script_content).encode('utf-8')
            
            # Create safe filename
            test_name = selected_case.get('name', 'unknown_test').replace(' ', '_').lower()
            filename = f"selenium_{test_name}.py"
            
            st.download_button(
                label="💾 Download Python Script",
                data=script_bytes,
                file_name=filename,
                mime="text/x-python",
                use_container_width=True
            )
        
        with col2:
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                # FIX: Copy the actual script content, not the dictionary
                if isinstance(script_content, str):
                    st.code(script_content, language='python')
                else:
                    st.code(str(script_content), language='python')
                st.success("Script content ready for copying!")
        
        # Execution instructions
        with st.expander("🚀 How to Execute This Script", expanded=True):
            st.markdown("""
            ### Prerequisites:
            1. **Install Selenium:** `pip install selenium`
            2. **Download ChromeDriver** and add to PATH
            3. **Update the URL** in the script to point to your application
            
            ### Execution:
            ```bash
            python selenium_script.py
            ```
            
            ### Expected Output:
            - Test execution logs
            - Success/failure status
            - Screenshot on failure (if configured)
            """)

def show_feature_analysis():
    st.header("🔍 Feature Analysis Report")
    
    if not st.session_state.documents_analyzed:
        st.warning("Please analyze documents first to see feature analysis")
        return
    
    st.success(f"📊 Analyzed {len(st.session_state.uploaded_files)} documents")
    
    # Feature overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Detected Features", len(st.session_state.detected_features))
    with col2:
        st.metric("Generated Test Cases", len(st.session_state.test_cases))
    with col3:
        test_cases_with_steps = len([tc for tc in st.session_state.test_cases if 'steps' in tc and tc['steps']])
        st.metric("Test Cases with Steps", test_cases_with_steps)
    
    # Detailed feature breakdown
    st.subheader("🎯 Detected Features Breakdown")
    
    if st.session_state.detected_features:
        for feature in st.session_state.detected_features:
            with st.expander(f"📋 {feature.replace('_', ' ').title()}", expanded=True):
                # Get test cases for this feature
                feature_tests = [tc for tc in st.session_state.test_cases if feature in tc.get('feature', '').lower()]
                
                st.write(f"**Test Cases:** {len(feature_tests)}")
                st.write(f"**Coverage:** {'✅ Good' if len(feature_tests) >= 2 else '⚠️ Limited'}")
                
                if feature_tests:
                    st.write("**Generated Tests:**")
                    for test in feature_tests[:3]:  # Show first 3
                        st.write(f"• {test.get('name', 'Unnamed Test')}")
                else:
                    st.info("No test cases generated for this feature yet")
    else:
        st.info("No features detected. Please analyze documents first.")

if __name__ == "__main__":
    main()
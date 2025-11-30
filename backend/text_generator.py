import logging
import re
import json
import math
from typing import List, Dict, Any, Optional
from knowledge_base import knowledge_base  # FIXED: Import the instance directly

logger = logging.getLogger(__name__)

class RAGEnhancedTestGenerator:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.test_case_counter = 1
        self.generation_patterns = {
            'positive': ['verify', 'validate', 'check', 'confirm', 'ensure'],
            'negative': ['invalid', 'error', 'incorrect', 'missing', 'wrong'],
            'boundary': ['empty', 'maximum', 'minimum', 'boundary', 'edge'],
            'workflow': ['complete', 'process', 'flow', 'scenario', 'journey']
        }
    
    def generate_test_cases(self, user_query: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Generate test cases using RAG and semantic context"""
        try:
            if context is None:
                context = {}
            
            logger.info(f"Generating test cases for query: {user_query}")
            
            # Perform semantic search for relevant content
            relevant_chunks = self.kb.semantic_search(user_query, top_k=5)
            
            # Get features and evidence from knowledge base
            available_features = self.kb.get_available_features()
            document_summary = self.kb.get_document_summary()
            
            # Analyze query intent
            query_intent = self._analyze_query_intent(user_query, available_features)
            
            # Generate test cases based on intent and available features
            test_cases = []
            
            if query_intent['type'] == 'specific_feature':
                test_cases.extend(self._generate_feature_specific_tests(
                    query_intent, available_features, relevant_chunks
                ))
            elif query_intent['type'] == 'workflow':
                test_cases.extend(self._generate_workflow_tests(
                    query_intent, document_summary, relevant_chunks
                ))
            else:
                test_cases.extend(self._generate_comprehensive_tests(
                    user_query, available_features, relevant_chunks, query_intent
                ))
            
            # If no test cases were generated, create fallback tests
            if not test_cases:
                test_cases.extend(self._generate_fallback_tests(user_query, available_features, relevant_chunks))
            
            # Enhance tests with RAG context
            enhanced_cases = []
            for test_case in test_cases:
                enhanced_case = self._enhance_with_rag_context(test_case, relevant_chunks)
                enhanced_case = self._add_evidence_and_selectors(enhanced_case)
                enhanced_case = self._calculate_confidence_score(enhanced_case, query_intent)
                enhanced_cases.append(enhanced_case)
            
            logger.info(f"Generated {len(enhanced_cases)} test cases with RAG context")
            return enhanced_cases
            
        except Exception as e:
            logger.error(f"Test generation error: {e}")
            return [self._create_error_test_case(user_query, str(e))]
    
    def _analyze_query_intent(self, query: str, available_features: List) -> Dict[str, Any]:
        """Analyze user query intent for targeted test generation"""
        query_lower = query.lower()
        
        intent = {
            "type": "general",
            "features": [],
            "workflows": [],
            "test_types": [],
            "confidence": 0.0
        }
        
        # Detect specific features - FIXED: Check against actual available features
        feature_names = [feature["name"].lower() for feature in available_features]
        
        # Check if query contains any of the available feature names
        for feature_name in feature_names:
            if feature_name in query_lower:
                intent["features"].append(feature_name)
                intent["type"] = "specific_feature"
                intent["confidence"] += 0.3
        
        # If no direct feature match, try keyword matching
        if not intent["features"]:
            feature_keywords = {
                'login_functionality': ['login', 'sign in', 'authentication', 'auth'],
                'checkout_process': ['checkout', 'purchase', 'payment', 'buy'],
                'search_functionality': ['search', 'find', 'query', 'lookup'],
                'form_submission': ['form', 'submit', 'input', 'field'],
                'profile_management': ['profile', 'account', 'settings', 'preferences'],
                'payment_processing': ['payment', 'credit card', 'billing', 'invoice']
            }
            
            for feature_name, keywords in feature_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    intent["features"].append(feature_name)
                    intent["type"] = "specific_feature"
                    intent["confidence"] += 0.3
                    break
        
        # Detect workflow requests
        workflow_indicators = ['workflow', 'process', 'flow', 'scenario', 'journey', 'end to end']
        if any(indicator in query_lower for indicator in workflow_indicators):
            intent["type"] = "workflow"
            intent["confidence"] += 0.4
        
        # Detect test types
        test_type_patterns = {
            'positive': ['positive', 'happy path', 'success', 'valid', 'normal'],
            'negative': ['negative', 'error', 'invalid', 'failure', 'wrong'],
            'boundary': ['boundary', 'edge', 'limit', 'maximum', 'minimum', 'empty'],
            'security': ['security', 'authentication', 'authorization', 'protected']
        }
        
        for test_type, patterns in test_type_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                intent["test_types"].append(test_type)
        
        # If no specific test types mentioned, include both positive and negative
        if not intent["test_types"]:
            intent["test_types"] = ["positive", "negative"]
        
        intent["confidence"] = min(intent["confidence"], 1.0)
        return intent
    
    def _generate_feature_specific_tests(self, intent: Dict, features: List, evidence: List[Dict]) -> List[Dict]:
        """Generate tests for specific features with evidence grounding"""
        test_cases = []
        
        for feature_name in intent["features"]:
            feature_data = next((f for f in features if f["name"] == feature_name), None)
            if not feature_data:
                # If feature not found in available features, create generic tests
                feature_data = {"name": feature_name, "confidence": 0.7}
            
            # Generate positive test cases
            if "positive" in intent["test_types"]:
                test_cases.extend(self._generate_positive_tests_for_feature(feature_name, feature_data, evidence))
            
            # Generate negative test cases
            if "negative" in intent["test_types"]:
                test_cases.extend(self._generate_negative_tests_for_feature(feature_name, feature_data, evidence))
            
            # Generate boundary test cases
            if "boundary" in intent["test_types"]:
                test_cases.extend(self._generate_boundary_tests_for_feature(feature_name, feature_data, evidence))
        
        return test_cases
    
    def _generate_positive_tests_for_feature(self, feature_name: str, feature_data: Dict, evidence: List[Dict]) -> List[Dict]:
        """Generate positive test cases for a specific feature"""
        tests = []
        
        if feature_name == "login_functionality":
            tests.extend([
                self._create_test_case(
                    test_id=self._get_next_id(),
                    name="Successful login with valid credentials",
                    description="Verify users can login with correct username and password",
                    category="Authentication",
                    priority="High",
                    steps=[
                        "Navigate to login page",
                        "Enter valid username in username field",
                        "Enter valid password in password field",
                        "Click login button",
                        "Verify successful redirect to dashboard",
                        "Verify user session is established"
                    ],
                    expected_result="User should be successfully logged in and redirected to dashboard",
                    feature=feature_name,
                    test_type="positive",
                    evidence=evidence
                )
            ])
        
        elif feature_name == "checkout_process":
            tests.extend([
                self._create_test_case(
                    test_id=self._get_next_id(),
                    name="Complete purchase with valid payment",
                    description="Verify complete checkout process with valid payment information",
                    category="E-commerce",
                    priority="High",
                    steps=[
                        "Add product to shopping cart",
                        "Navigate to checkout page",
                        "Fill shipping address information",
                        "Select shipping method",
                        "Enter valid payment details",
                        "Click place order button",
                        "Verify order confirmation page",
                        "Verify order details are correct"
                    ],
                    expected_result="Order should be successfully placed with confirmation",
                    feature=feature_name,
                    test_type="positive",
                    evidence=evidence
                )
            ])
        
        elif feature_name == "search_functionality":
            tests.extend([
                self._create_test_case(
                    test_id=self._get_next_id(),
                    name="Search with valid query returns results",
                    description="Verify search functionality returns relevant results",
                    category="Search",
                    priority="High",
                    steps=[
                        "Navigate to search page",
                        "Enter valid search term in search field",
                        "Click search button or press enter",
                        "Verify search results are displayed",
                        "Verify results are relevant to search term",
                        "Verify result count is shown"
                    ],
                    expected_result="Relevant search results should be displayed",
                    feature=feature_name,
                    test_type="positive",
                    evidence=evidence
                )
            ])
        
        elif feature_name == "form_submission":
            tests.extend([
                self._create_test_case(
                    test_id=self._get_next_id(),
                    name="Successful form submission with valid data",
                    description="Verify form can be submitted with all required fields filled correctly",
                    category="Forms",
                    priority="High",
                    steps=[
                        "Navigate to form page",
                        "Fill all required form fields with valid data",
                        "Verify field validation passes",
                        "Click submit button",
                        "Verify form submission success message",
                        "Verify user is redirected to confirmation page"
                    ],
                    expected_result="Form should be submitted successfully with proper confirmation",
                    feature=feature_name,
                    test_type="positive",
                    evidence=evidence
                )
            ])
        
        elif feature_name == "profile_management":
            tests.extend([
                self._create_test_case(
                    test_id=self._get_next_id(),
                    name="Update user profile information",
                    description="Verify users can update their profile details",
                    category="User Management",
                    priority="Medium",
                    steps=[
                        "Navigate to profile page",
                        "Update profile information (name, email, preferences)",
                        "Save profile changes",
                        "Verify success message is displayed",
                        "Verify changes are persisted",
                        "Refresh page and verify changes are still present"
                    ],
                    expected_result="Profile information should be updated successfully",
                    feature=feature_name,
                    test_type="positive",
                    evidence=evidence
                )
            ])
        
        else:
            # Generic positive test for any feature
            tests.append(
                self._create_test_case(
                    test_id=self._get_next_id(),
                    name=f"Verify {feature_name.replace('_', ' ')} functionality",
                    description=f"Positive test for {feature_name.replace('_', ' ')} based on documentation",
                    category=feature_name.split('_')[0].title() if '_' in feature_name else "General",
                    priority="High",
                    steps=self._generate_steps_for_feature(feature_name, "positive"),
                    expected_result=f"{feature_name.replace('_', ' ')} should work as specified",
                    feature=feature_name,
                    test_type="positive",
                    evidence=evidence
                )
            )
        
        return tests
    
    def _generate_negative_tests_for_feature(self, feature_name: str, feature_data: Dict, evidence: List[Dict]) -> List[Dict]:
        """Generate negative test cases for a specific feature"""
        tests = []
        
        if feature_name == "login_functionality":
            tests.extend([
                self._create_test_case(
                    test_id=self._get_next_id(),
                    name="Login failure with invalid credentials",
                    description="Verify appropriate error for invalid login attempts",
                    category="Authentication",
                    priority="High",
                    steps=[
                        "Navigate to login page",
                        "Enter invalid username",
                        "Enter invalid password",
                        "Click login button",
                        "Verify error message is displayed",
                        "Verify user remains on login page"
                    ],
                    expected_result="Appropriate error message should be shown for invalid credentials",
                    feature=feature_name,
                    test_type="negative",
                    evidence=evidence
                ),
                self._create_test_case(
                    test_id=self._get_next_id(),
                    name="Login attempt with empty credentials",
                    description="Verify validation for empty required fields",
                    category="Authentication",
                    priority="Medium",
                    steps=[
                        "Navigate to login page",
                        "Leave username field empty",
                        "Leave password field empty",
                        "Click login button",
                        "Verify validation messages are shown"
                    ],
                    expected_result="Validation messages should appear for empty required fields",
                    feature=feature_name,
                    test_type="negative",
                    evidence=evidence
                )
            ])
        
        elif feature_name == "checkout_process":
            tests.extend([
                self._create_test_case(
                    test_id=self._get_next_id(),
                    name="Checkout failure with invalid payment",
                    description="Verify error handling for invalid payment information",
                    category="E-commerce",
                    priority="High",
                    steps=[
                        "Add product to shopping cart",
                        "Navigate to checkout page",
                        "Fill valid shipping information",
                        "Enter invalid payment details",
                        "Click place order button",
                        "Verify payment error message",
                        "Verify order is not processed"
                    ],
                    expected_result="Appropriate payment error should prevent order processing",
                    feature=feature_name,
                    test_type="negative",
                    evidence=evidence
                )
            ])
        
        elif feature_name == "form_submission":
            tests.extend([
                self._create_test_case(
                    test_id=self._get_next_id(),
                    name="Form submission with missing required fields",
                    description="Verify form validation for required fields",
                    category="Forms",
                    priority="High",
                    steps=[
                        "Navigate to form page",
                        "Leave required fields empty",
                        "Click submit button",
                        "Verify validation error messages",
                        "Verify form is not submitted"
                    ],
                    expected_result="Validation errors should prevent form submission",
                    feature=feature_name,
                    test_type="negative",
                    evidence=evidence
                )
            ])
        
        else:
            # Generic negative test for any feature
            tests.append(
                self._create_test_case(
                    test_id=self._get_next_id(),
                    name=f"Verify {feature_name.replace('_', ' ')} error handling",
                    description=f"Negative test for {feature_name.replace('_', ' ')} error conditions",
                    category=feature_name.split('_')[0].title() if '_' in feature_name else "General",
                    priority="Medium",
                    steps=self._generate_steps_for_feature(feature_name, "negative"),
                    expected_result=f"Appropriate errors should be handled for {feature_name.replace('_', ' ')}",
                    feature=feature_name,
                    test_type="negative",
                    evidence=evidence
                )
            )
        
        return tests
    
    def _generate_boundary_tests_for_feature(self, feature_name: str, feature_data: Dict, evidence: List[Dict]) -> List[Dict]:
        """Generate boundary test cases for a specific feature"""
        tests = []
        
        if feature_name == "form_submission":
            tests.extend([
                self._create_test_case(
                    test_id=self._get_next_id(),
                    name="Form submission with maximum field lengths",
                    description="Verify form handles maximum input lengths correctly",
                    category="Forms",
                    priority="Medium",
                    steps=[
                        "Navigate to form page",
                        "Fill fields with maximum allowed characters",
                        "Verify no validation errors for max length",
                        "Submit form",
                        "Verify successful submission"
                    ],
                    expected_result="Form should accept maximum length inputs without errors",
                    feature=feature_name,
                    test_type="boundary",
                    evidence=evidence
                )
            ])
        
        return tests
    
    def _generate_workflow_tests(self, intent: Dict, summary: Dict, evidence: List[Dict]) -> List[Dict]:
        """Generate comprehensive workflow test cases"""
        tests = []
        
        # Extract workflows from documents
        all_workflows = []
        for doc_type, doc_data in self.kb.documents.items():
            all_workflows.extend(doc_data.get("workflows", []))
        
        for workflow in all_workflows[:3]:  # Limit to top 3 workflows
            tests.append(
                self._create_test_case(
                    test_id=self._get_next_id(),
                    name=f"End-to-end {workflow['type']} workflow",
                    description=f"Complete workflow: {workflow['description'][:100]}...",
                    category="Workflow",
                    priority="High",
                    steps=self._convert_workflow_to_steps(workflow),
                    expected_result=f"Complete {workflow['type']} workflow should execute successfully",
                    feature=workflow['type'],
                    test_type="workflow",
                    evidence=evidence
                )
            )
        
        return tests
    
    def _generate_comprehensive_tests(self, query: str, features: List, evidence: List[Dict], intent: Dict) -> List[Dict]:
        """Generate comprehensive test coverage based on available features"""
        tests = []
        
        # Generate tests for high-confidence features
        high_confidence_features = [f for f in features if f.get('confidence', 0) > 0.7]
        
        for feature in high_confidence_features[:5]:  # Limit to top 5 features
            feature_name = feature["name"]
            
            # Positive test
            if "positive" in intent["test_types"]:
                tests.append(
                    self._create_test_case(
                        test_id=self._get_next_id(),
                        name=f"Verify {feature_name.replace('_', ' ')} functionality",
                        description=f"Positive test for {feature_name.replace('_', ' ')} based on documentation",
                        category=feature_name.split('_')[0].title() if '_' in feature_name else "General",
                        priority="High",
                        steps=self._generate_steps_for_feature(feature_name, "positive"),
                        expected_result=f"{feature_name.replace('_', ' ')} should work as specified",
                        feature=feature_name,
                        test_type="positive",
                        evidence=evidence
                    )
                )
            
            # Negative test
            if "negative" in intent["test_types"]:
                tests.append(
                    self._create_test_case(
                        test_id=self._get_next_id(),
                        name=f"Verify {feature_name.replace('_', ' ')} error handling",
                        description=f"Negative test for {feature_name.replace('_', ' ')} error conditions",
                        category=feature_name.split('_')[0].title() if '_' in feature_name else "General",
                        priority="Medium",
                        steps=self._generate_steps_for_feature(feature_name, "negative"),
                        expected_result=f"Appropriate errors should be handled for {feature_name.replace('_', ' ')}",
                        feature=feature_name,
                        test_type="negative",
                        evidence=evidence
                    )
                )
        
        return tests
    
    def _generate_fallback_tests(self, query: str, features: List, evidence: List[Dict]) -> List[Dict]:
        """Generate fallback test cases when no specific features are detected"""
        tests = []
        
        # Get top 3 features by confidence
        top_features = sorted(features, key=lambda x: x.get('confidence', 0), reverse=True)[:3]
        
        for feature in top_features:
            feature_name = feature["name"]
            
            # Generate both positive and negative tests
            tests.extend([
                self._create_test_case(
                    test_id=self._get_next_id(),
                    name=f"Verify {feature_name.replace('_', ' ')} functionality",
                    description=f"Test {feature_name.replace('_', ' ')} based on query: {query}",
                    category=feature_name.split('_')[0].title() if '_' in feature_name else "General",
                    priority="High",
                    steps=self._generate_steps_for_feature(feature_name, "positive"),
                    expected_result=f"{feature_name.replace('_', ' ')} should work correctly",
                    feature=feature_name,
                    test_type="positive",
                    evidence=evidence
                ),
                self._create_test_case(
                    test_id=self._get_next_id(),
                    name=f"Verify {feature_name.replace('_', ' ')} error handling",
                    description=f"Test {feature_name.replace('_', ' ')} error scenarios based on query: {query}",
                    category=feature_name.split('_')[0].title() if '_' in feature_name else "General",
                    priority="Medium",
                    steps=self._generate_steps_for_feature(feature_name, "negative"),
                    expected_result=f"Appropriate error handling for {feature_name.replace('_', ' ')}",
                    feature=feature_name,
                    test_type="negative",
                    evidence=evidence
                )
            ])
        
        return tests
    
    def _create_test_case(self, test_id: str, name: str, description: str, category: str, 
                         priority: str, steps: List[str], expected_result: str,
                         feature: str, test_type: str, evidence: List[Dict]) -> Dict[str, Any]:
        """Create a standardized test case structure"""
        return {
            "id": f"TC-{test_id}",
            "name": name,
            "description": description,
            "category": category,
            "priority": priority,
            "steps": steps,
            "expected_result": expected_result,
            "feature": feature,
            "test_type": test_type,
            "tags": [feature, test_type, category.lower()],
            "test_data": self._generate_test_data(feature, test_type),
            "source_evidence": [chunk.get('text', '') for chunk in evidence[:2]],
            "generation_context": {
                "feature_confidence": next((f.get('confidence', 0) for f in self.kb.get_available_features() if f['name'] == feature), 0),
                "evidence_count": len(evidence),
                "test_type": test_type
            }
        }
    
    def _enhance_with_rag_context(self, test_case: Dict, evidence: List[Dict]) -> Dict:
        """Enhance test case with RAG context and evidence"""
        if evidence:
            # Add relevant evidence snippets
            test_case["rag_evidence"] = [
                {
                    "text": chunk.get('text', ''),
                    "relevance_score": chunk.get('similarity', 0.5),
                    "source": chunk.get('doc_type', 'unknown')
                }
                for chunk in evidence[:3]
            ]
            
            # Enhance steps based on evidence
            enhanced_steps = self._enhance_steps_with_evidence(test_case["steps"], evidence)
            if enhanced_steps:
                test_case["enhanced_steps"] = enhanced_steps
        
        return test_case
    
    def _enhance_steps_with_evidence(self, steps: List[str], evidence: List[Dict]) -> List[str]:
        """Enhance test steps with evidence from RAG context"""
        enhanced_steps = []
        
        for step in steps:
            enhanced_step = step
            
            # Look for relevant evidence for this step
            for chunk in evidence:
                chunk_text = chunk.get('text', '').lower()
                step_lower = step.lower()
                
                # If evidence contains details about this step, enhance it
                if any(keyword in chunk_text for keyword in step_lower.split()[:3]):
                    # Extract additional details from evidence
                    details = self._extract_step_details_from_evidence(step, chunk_text)
                    if details:
                        enhanced_step = f"{step} {details}"
                        break
            
            enhanced_steps.append(enhanced_step)
        
        return enhanced_steps if enhanced_steps != steps else []
    
    def _extract_step_details_from_evidence(self, step: str, evidence_text: str) -> str:
        """Extract additional details for test steps from evidence"""
        step_lower = step.lower()
        
        if 'enter' in step_lower and 'field' in step_lower:
            # Look for field-specific validation in evidence
            validation_patterns = [
                r'must be (\d+) characters',
                r'required field',
                r'must contain ([a-zA-Z]+)',
                r'format should be ([^.!?]+)'
            ]
            
            for pattern in validation_patterns:
                match = re.search(pattern, evidence_text)
                if match:
                    return f"(Validation: {match.group(0)})"
        
        elif 'click' in step_lower and 'button' in step_lower:
            # Look for button behavior in evidence
            behavior_patterns = [
                r'redirects to ([^.!?]+)',
                r'opens ([^.!?]+)',
                r'displays ([^.!?]+)'
            ]
            
            for pattern in behavior_patterns:
                match = re.search(pattern, evidence_text)
                if match:
                    return f"(Expected: {match.group(0)})"
        
        return ""
    
    def _add_evidence_and_selectors(self, test_case: Dict) -> Dict:
        """Add evidence references and selector mappings to test case"""
        # Extract elements from test steps
        elements = set()
        for step in test_case.get("steps", []):
            extracted_elements = self._extract_elements_from_step(step)
            elements.update(extracted_elements)
        
        # Map elements to selectors
        selector_mappings = {}
        for element in elements:
            selectors = self.kb.map_to_selectors(element)
            if selectors:
                selector_mappings[element] = selectors
        
        if selector_mappings:
            test_case["selector_mappings"] = selector_mappings
        
        # Add feature evidence
        feature_name = test_case.get("feature")
        if feature_name:
            feature_evidence = self._get_feature_evidence(feature_name)
            if feature_evidence:
                test_case["feature_evidence"] = feature_evidence
        
        return test_case
    
    def _calculate_confidence_score(self, test_case: Dict, intent: Dict) -> Dict:
        """Calculate confidence score for test case quality"""
        confidence = 0.0
        
        # Base confidence from intent
        confidence += intent.get("confidence", 0.0) * 0.3
        
        # Evidence-based scoring
        if test_case.get("source_evidence"):
            confidence += 0.2
            if len(test_case["source_evidence"]) > 1:
                confidence += 0.1
        
        if test_case.get("rag_evidence"):
            confidence += 0.1
        
        # Selector mapping scoring
        if test_case.get("selector_mappings"):
            confidence += 0.1
        
        # Step detail scoring
        steps = test_case.get("steps", [])
        if len(steps) >= 4:
            confidence += 0.1
        if len(steps) >= 6:
            confidence += 0.1
        
        # Test data scoring
        if test_case.get("test_data"):
            confidence += 0.1
        
        test_case["confidence_score"] = round(min(confidence, 1.0), 2)
        test_case["quality_indicators"] = {
            "well_grounded": confidence > 0.7,
            "has_evidence": bool(test_case.get("source_evidence")),
            "has_selectors": bool(test_case.get("selector_mappings")),
            "step_completeness": len(steps) >= 4
        }
        
        return test_case
    
    def _extract_elements_from_step(self, step: str) -> List[str]:
        """Extract UI elements from test step"""
        elements = []
        step_lower = step.lower()
        
        patterns = [
            r'enter[^.!?]*?in ([^.!?]+?) field',
            r'click ([^.!?]+?) button',
            r'select ([^.!?]+?) from',
            r'fill ([^.!?]+?) field',
            r'verify ([^.!?]+?) is displayed',
            r'check ([^.!?]+?) is present'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, step_lower)
            for match in matches:
                element = match.strip()
                if element and element not in ['the', 'a', 'an']:
                    elements.append(element)
        
        return elements
    
    def _get_feature_evidence(self, feature_name: str) -> List[str]:
        """Get evidence snippets for a specific feature"""
        evidence = []
        for doc_type, doc_data in self.kb.documents.items():
            features = doc_data.get("features", [])
            for feature in features:
                if feature.get("name") == feature_name:
                    evidence.extend(feature.get("evidence", [])[:2])
                    break
        return evidence
    
    def _convert_workflow_to_steps(self, workflow: Dict) -> List[str]:
        """Convert workflow definition to test steps"""
        steps = []
        for workflow_step in workflow.get("steps", []):
            action = workflow_step.get("action", "perform")
            target = workflow_step.get("target", "element")
            data = workflow_step.get("data")
            
            step_description = f"{action} {target}"
            if data:
                step_description += f" with {data}"
            
            steps.append(step_description.capitalize())
        
        return steps if steps else ["Execute complete workflow based on documentation"]
    
    def _generate_steps_for_feature(self, feature_name: str, test_type: str) -> List[str]:
        """Generate appropriate steps for a feature and test type"""
        if feature_name == "login_functionality":
            if test_type == "positive":
                return [
                    "Navigate to login page",
                    "Enter valid username",
                    "Enter valid password", 
                    "Click login button",
                    "Verify successful authentication",
                    "Verify redirect to dashboard"
                ]
            else:  # negative
                return [
                    "Navigate to login page",
                    "Enter invalid credentials",
                    "Click login button", 
                    "Verify error message display",
                    "Verify no redirect occurs"
                ]
        
        elif feature_name == "search_functionality":
            return [
                "Navigate to search interface",
                "Enter search query",
                "Execute search",
                "Verify results display",
                "Verify result relevance"
            ]
        
        elif feature_name == "form_submission":
            if test_type == "positive":
                return [
                    "Navigate to form page",
                    "Fill all required fields with valid data",
                    "Verify field validation passes",
                    "Click submit button",
                    "Verify submission success",
                    "Verify confirmation message"
                ]
            else:  # negative
                return [
                    "Navigate to form page",
                    "Leave required fields empty",
                    "Click submit button",
                    "Verify validation errors",
                    "Verify form not submitted"
                ]
        
        elif feature_name == "profile_management":
            return [
                "Navigate to profile page",
                "Update profile information",
                "Save changes",
                "Verify success message",
                "Verify changes persisted"
            ]
        
        # Default steps for unknown features
        return [
            f"Navigate to {feature_name.replace('_', ' ')} page",
            f"Perform {feature_name.replace('_', ' ')} actions",
            f"Verify {feature_name.replace('_', ' ')} results",
            f"Validate {feature_name.replace('_', ' ')} behavior"
        ]
    
    def _generate_test_data(self, feature: str, test_type: str) -> Dict[str, Any]:
        """Generate appropriate test data for feature and test type"""
        test_data = {}
        
        if feature == "login_functionality":
            if test_type == "positive":
                test_data = {
                    "username": "test_user@example.com",
                    "password": "SecurePassword123!"
                }
            else:
                test_data = {
                    "username": "invalid_user",
                    "password": "wrong_password"
                }
        
        elif feature == "checkout_process":
            test_data = {
                "shipping_address": "123 Test Street, Test City, 12345",
                "payment_card": "4111111111111111",
                "expiry_date": "12/25",
                "cvv": "123"
            }
        
        elif feature_name == "search_functionality":
            test_data = {
                "search_query": "test product",
                "expected_results_min": 1
            }
        
        elif feature_name == "form_submission":
            test_data = {
                "sample_data": "Test form data",
                "validation_rules": "Check required fields"
            }
        
        return test_data
    
    def _get_next_id(self) -> str:
        """Get next test case ID"""
        current_id = self.test_case_counter
        self.test_case_counter += 1
        return str(current_id).zfill(3)
    
    def _create_error_test_case(self, user_query: str, error_msg: str) -> Dict[str, Any]:
        """Create error test case when generation fails"""
        return {
            "id": "TC-ERROR-001",
            "name": f"Error generating test for: {user_query}",
            "description": f"Test generation failed: {error_msg}",
            "steps": [
                "Check system logs for detailed error information",
                "Verify document parsing is working correctly",
                "Retry test generation with different query"
            ],
            "expected_result": "System should generate appropriate test cases without errors",
            "priority": "High",
            "category": "System",
            "confidence_score": 0.0,
            "error": True
        }

# Initialize enhanced test generator
enhanced_test_generator = RAGEnhancedTestGenerator(knowledge_base)
import json
import os
import re
import logging
import math
import io
from typing import Dict, List, Any, Optional, Union
from bs4 import BeautifulSoup
import PyPDF2

logger = logging.getLogger(__name__)

class RAGEnhancedKnowledgeBase:
    def __init__(self):
        self.documents = {}
        self.semantic_index = {}
        self.text_chunks = []
        self.chunk_embeddings = []
        
        # Enhanced selector patterns with context awareness
        self.selector_patterns = {
            'username': ['#username', '.username-input', '[name="username"]', 'input[type="text"]:first-of-type'],
            'password': ['#password', '.password-field', '[type="password"]', '[name="password"]'],
            'email': ['#email', '[name="email"]', '.email-field', 'input[type="email"]'],
            'submit': ['#submit', '.btn-submit', '[type="submit"]', 'button[type="submit"]', 'input[value*="Submit"]'],
            'login': ['#login', '.login-btn', '[value*="Login"]', 'button:contains("Login")'],
            'search': ['#search', '.search-input', '[name="query"]', 'input[type="search"]'],
            'cart': ['#cart', '.cart-btn', '.shopping-cart', '[href*="cart"]'],
            'checkout': ['#checkout', '.checkout-btn', '[value*="Checkout"]', 'button:contains("Checkout")'],
            'first_name': ['#firstName', '[name="first_name"]', '.first-name'],
            'last_name': ['#lastName', '[name="last_name"]', '.last-name'],
            'address': ['#address', '[name="address"]', '.address-field'],
            'phone': ['#phone', '[name="phone"]', '.phone-number', 'input[type="tel"]']
        }
        
        # Workflow patterns for dynamic test generation
        self.workflow_patterns = {
            'login': ['login', 'sign in', 'authenticate', 'credentials'],
            'checkout': ['checkout', 'purchase', 'buy now', 'place order', 'payment'],
            'search': ['search', 'find', 'query', 'lookup'],
            'registration': ['register', 'sign up', 'create account', 'join'],
            'profile': ['profile', 'account', 'settings', 'preferences'],
            'navigation': ['navigate', 'menu', 'home', 'categories', 'browse']
        }
        
        self._load_existing_data()

    def _load_existing_data(self):
        """Load and parse existing documentation files"""
        try:
            data_files = {
                'product_specs': 'data/product_specs.md',
                'ui_ux_guide': 'data/ui_ux_guide.txt',
                'checkout_html': 'data/checkout.html',
                'api_endpoints': 'data/api_endpoints.json'
            }
            
            for doc_type, file_path in data_files.items():
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        self.parse_document(content, doc_type)
                        logger.info(f"Loaded and parsed {doc_type}")
                        
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")

    def parse_document(self, content: Union[str, bytes], doc_type: str) -> Dict[str, Any]:
        """Enhanced document parsing with semantic chunking and RAG preparation"""
        
        # FIX: Extract text from PDF if content is bytes and type is pdf
        text_content = ""
        if doc_type == 'pdf' and isinstance(content, bytes):
            text_content = self._extract_text_from_pdf(content)
        elif isinstance(content, bytes):
            # Fallback decode for other binary types
            text_content = content.decode('utf-8', errors='ignore')
        else:
            text_content = str(content)

        parsed_data = {
            "features": [],
            "workflows": [],
            "validation_rules": [],
            "ui_elements": [],
            "api_endpoints": [],
            "semantic_chunks": [],
            "extracted_selectors": [],
            "evidence_snippets": [],
            "raw_content": text_content[:1000] + "...", # Store snippet only to save memory
            "doc_type": doc_type
        }
        
        # Extract features using enhanced pattern matching
        parsed_data["features"] = self._extract_features(text_content)
        
        # Extract workflows with context
        parsed_data["workflows"] = self._extract_workflows(text_content)
        
        # Extract validation rules
        parsed_data["validation_rules"] = self._extract_validation_rules(text_content)
        
        # Extract UI elements and selectors
        parsed_data["ui_elements"] = self._extract_ui_elements(text_content)
        parsed_data["extracted_selectors"] = self._extract_selectors_from_content(text_content)
        
        # Create semantic chunks for RAG
        chunks = self._create_semantic_chunks(text_content)
        parsed_data["semantic_chunks"] = chunks
        
        # Extract evidence snippets
        parsed_data["evidence_snippets"] = self._extract_evidence_snippets(text_content)
        
        # Build semantic index
        for chunk in chunks:
            self._add_to_semantic_index(chunk, doc_type)
        
        # Store document
        self.documents[doc_type] = parsed_data
        
        return parsed_data

    def _extract_text_from_pdf(self, content: bytes) -> str:
        """Helper to extract text from PDF bytes using PyPDF2"""
        try:
            text = ""
            pdf_file = io.BytesIO(content)
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            return ""

    def _extract_features(self, content: str) -> List[Dict[str, Any]]:
        """Extract features with context and confidence scoring"""
        features = []
        content_lower = content.lower()
        
        feature_patterns = {
            'login_functionality': ['login', 'sign in', 'authenticate', 'credentials', 'username', 'password'],
            'checkout_process': ['checkout', 'purchase', 'buy now', 'place order', 'shopping cart', 'payment'],
            'search_functionality': ['search', 'find', 'query', 'lookup', 'filter', 'sort'],
            'form_submission': ['form', 'submit', 'input', 'field', 'validation'],
            'shopping_cart': ['cart', 'basket', 'add to cart', 'remove from cart'],
            'payment_processing': ['payment', 'credit card', 'paypal', 'billing', 'invoice'],
            'user_registration': ['register', 'sign up', 'create account', 'join'],
            'profile_management': ['profile', 'account', 'settings', 'preferences']
        }
        
        for feature_name, keywords in feature_patterns.items():
            matches = []
            for keyword in keywords:
                if keyword in content_lower:
                    # Find context around the keyword
                    context = self._extract_keyword_context(content, keyword)
                    matches.extend(context)
            
            if matches:
                confidence = min(len(matches) * 0.2, 1.0)  # Confidence based on occurrences
                features.append({
                    "name": feature_name,
                    "confidence": round(confidence, 2),
                    "evidence": matches[:3],  # Top 3 evidence snippets
                    "keyword_matches": len(matches)
                })
        
        return features

    def _extract_workflows(self, content: str) -> List[Dict[str, Any]]:
        """Extract user workflows with step-by-step context"""
        workflows = []
        
        # Pattern for multi-step workflows
        workflow_patterns = [
            r'(?:first|then|next|after that|finally)[^.!?]*?(?:click|enter|select|navigate|submit)[^.!?]*?[.!?]',
            r'(?:step \d+[^.!?]*?)(?:click|enter|select|navigate|submit)[^.!?]*?[.!?]',
            r'(?:to[^.!?]*?)(?:click|enter|select|navigate|submit)[^.!?]*?(?:then|and|or)[^.!?]*?[.!?]'
        ]
        
        for pattern in workflow_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                workflow_text = match.group(0)
                steps = self._parse_workflow_steps(workflow_text)
                if steps:
                    workflows.append({
                        "description": workflow_text,
                        "steps": steps,
                        "step_count": len(steps),
                        "type": self._classify_workflow(workflow_text)
                    })
        
        return workflows

    def _parse_workflow_steps(self, workflow_text: str) -> List[Dict[str, str]]:
        """Parse individual steps from workflow description"""
        steps = []
        step_indicators = ['first', 'then', 'next', 'after that', 'finally', 'step 1', 'step 2', 'step 3']
        
        # Split by step indicators
        current_step = ""
        for sentence in re.split(r'[.!?]+', workflow_text):
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in step_indicators):
                if current_step:
                    step_data = self._analyze_step(current_step)
                    if step_data:
                        steps.append(step_data)
                current_step = sentence
            elif current_step:
                current_step += ". " + sentence
        
        if current_step:
            step_data = self._analyze_step(current_step)
            if step_data:
                steps.append(step_data)
        
        return steps

    def _analyze_step(self, step_text: str) -> Dict[str, str]:
        """Analyze a single workflow step"""
        step_lower = step_text.lower()
        
        action = "perform"
        target = "element"
        data = None
        
        # Determine action type
        if any(word in step_lower for word in ['click', 'press', 'tap']):
            action = "click"
        elif any(word in step_lower for word in ['enter', 'type', 'input', 'fill']):
            action = "enter"
        elif any(word in step_lower for word in ['select', 'choose', 'pick']):
            action = "select"
        elif any(word in step_lower for word in ['navigate', 'go to', 'visit']):
            action = "navigate"
        elif any(word in step_lower for word in ['verify', 'check', 'confirm']):
            action = "verify"
        
        # Extract target element
        target_patterns = [
            r'(?:click|press|tap)\s+(?:on\s+)?(?:the\s+)?([^.!?,]+?)\s+(?:button|link|icon)',
            r'(?:enter|type|input|fill)\s+(?:.*?)\s+in\s+(?:the\s+)?([^.!?,]+?)\s+(?:field|input|box)',
            r'(?:select|choose)\s+(?:.*?)\s+from\s+(?:the\s+)?([^.!?,]+?)\s+(?:dropdown|menu|list)'
        ]
        
        for pattern in target_patterns:
            match = re.search(pattern, step_lower)
            if match:
                target = match.group(1).strip()
                break
        
        # Extract data for enter actions
        if action == "enter":
            data_match = re.search(r'(?:enter|type|input|fill)\s+([^.!?]+?)\s+(?:in|into)', step_lower)
            if data_match:
                data = data_match.group(1).strip()
        
        return {
            "action": action,
            "target": target,
            "data": data,
            "description": step_text,
            "suggested_selectors": self.map_to_selectors(target)
        }

    def _extract_validation_rules(self, content: str) -> List[Dict[str, Any]]:
        """Extract validation rules with context"""
        validation_rules = []
        
        validation_patterns = {
            'required': r'(?:must|should)\s+be\s+provided|required\s+field|cannot\s+be\s+empty',
            'format': r'(?:must|should)\s+be\s+(?:a|an)\s+[^.!?]*?(?:email|phone|number|date)',
            'length': r'(?:must|should)\s+be\s+(?:at least|at most|between)[^.!?]*?\d+',
            'range': r'(?:must|should)\s+be\s+(?:between|from)[^.!?]*?\d+[^.!?]*?\d+'
        }
        
        for rule_type, pattern in validation_patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                context = self._extract_keyword_context(content, match.group(0))
                validation_rules.append({
                    "type": rule_type,
                    "rule": match.group(0),
                    "context": context[0] if context else match.group(0),
                    "field": self._extract_field_from_context(match.group(0))
                })
        
        return validation_rules

    def _extract_ui_elements(self, content: str) -> List[Dict[str, Any]]:
        """Extract UI elements with enhanced context"""
        ui_elements = []
        
        element_patterns = {
            'button': r'button[^>]*>.*?</button>|input[^>]*type=["\'](submit|button)["\']',
            'input': r'input[^>]*type=["\'](text|email|password|number)["\'][^>]*>',
            'dropdown': r'<select[^>]*>.*?</select>',
            'checkbox': r'input[^>]*type=["\'](checkbox|radio)["\'][^>]*>',
            'link': r'<a[^>]*href=["\'][^"^\']*["\'][^>]*>.*?</a>'
        }
        
        for element_type, pattern in element_patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                element_html = match.group(0)
                attributes = self._extract_attributes(element_html)
                ui_elements.append({
                    "type": element_type,
                    "html": element_html,
                    "attributes": attributes,
                    "suggested_selectors": self._generate_selectors_from_html(element_html, element_type),
                    "context": self._extract_html_context(content, element_html)
                })
        
        return ui_elements

    def _extract_selectors_from_content(self, content: str) -> List[Dict[str, Any]]:
        """Extract CSS selectors from HTML content"""
        selectors = []
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract IDs
            for element in soup.find_all(id=True):
                selectors.append({
                    "type": "id",
                    "value": f"#{element['id']}",
                    "element": element.name,
                    "context": str(element)[:100]
                })
            
            # Extract classes
            for element in soup.find_all(class_=True):
                for class_name in element.get('class', []):
                    selectors.append({
                        "type": "class", 
                        "value": f".{class_name}",
                        "element": element.name,
                        "context": str(element)[:100]
                    })
            
            # Extract form elements
            for element in soup.find_all(['input', 'select', 'textarea']):
                name = element.get('name')
                if name:
                    selectors.append({
                        "type": "name",
                        "value": f'[name="{name}"]',
                        "element": element.name,
                        "input_type": element.get('type', ''),
                        "context": str(element)[:100]
                    })
                    
        except Exception as e:
            logger.error(f"Error parsing HTML for selectors: {e}")
        
        return selectors

    def _create_semantic_chunks(self, content: str, chunk_size: int = 150) -> List[Dict[str, Any]]:
        """Create semantic chunks for RAG with overlapping context"""
        sentences = re.split(r'[.!?]+', content)
        chunks = []
        current_chunk = ""
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    embedding = self._simple_embedding(current_chunk)
                    chunks.append({
                        "text": current_chunk.strip(),
                        "embedding": embedding,
                        "token_count": len(current_chunk.split()),
                        "chunk_id": f"chunk_{len(chunks)}"
                    })
                current_chunk = sentence + ". "
        
        if current_chunk:
            embedding = self._simple_embedding(current_chunk)
            chunks.append({
                "text": current_chunk.strip(),
                "embedding": embedding,
                "token_count": len(current_chunk.split()),
                "chunk_id": f"chunk_{len(chunks)}"
            })
        
        return chunks

    def _simple_embedding(self, text: str) -> List[float]:
        """Simple TF-based embedding without external dependencies"""
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Create a simple numerical representation
        embedding = [len(text), len(words), sum(word_freq.values())]
        
        # Add some basic semantic features
        features = {
            'has_form': 1.0 if any(word in text.lower() for word in ['form', 'input', 'submit']) else 0.0,
            'has_button': 1.0 if 'button' in text.lower() else 0.0,
            'has_validation': 1.0 if any(word in text.lower() for word in ['required', 'validate', 'check']) else 0.0,
            'has_navigation': 1.0 if any(word in text.lower() for word in ['click', 'navigate', 'go to']) else 0.0
        }
        
        embedding.extend(features.values())
        return embedding

    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Semantic search using simple cosine similarity"""
        if not self.text_chunks:
            return []
        
        query_embedding = self._simple_embedding(query)
        similarities = []
        
        for chunk in self.text_chunks:
            similarity = self._cosine_similarity(query_embedding, chunk['embedding'])
            similarities.append((similarity, chunk))
        
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [item for _, item in similarities[:top_k]]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def map_to_selectors(self, element_description: str) -> List[str]:
        """Enhanced selector mapping with context awareness"""
        description_lower = element_description.lower()
        selectors = []
        
        # Direct keyword mapping
        for key, selector_list in self.selector_patterns.items():
            if key in description_lower:
                selectors.extend(selector_list)
        
        # Context-based mapping
        if 'button' in description_lower:
            button_text = self._extract_button_text(description_lower)
            if button_text:
                selectors.extend([
                    f'//button[contains(text(), "{button_text}")]',
                    f'//input[@value="{button_text}"]',
                    f'button:contains("{button_text}")'
                ])
        
        if any(word in description_lower for word in ['input', 'field', 'text']):
            field_name = self._extract_field_name(description_lower)
            if field_name:
                selectors.extend([
                    f'[name="{field_name}"]',
                    f'[id="{field_name}"]',
                    f'.{field_name.replace(" ", "-")}-field'
                ])
        
        # Add generic selectors as fallback
        if not selectors:
            clean_description = re.sub(r'[^a-zA-Z0-9]', '-', element_description.lower())
            selectors.extend([
                f'[data-testid="{clean_description}"]',
                f'[aria-label*="{element_description}"]',
                f'//*[contains(text(), "{element_description}")]'
            ])
        
        return list(set(selectors))  # Remove duplicates

    def _extract_button_text(self, text: str) -> str:
        """Extract button text from description"""
        patterns = [
            r'["\']([^"^\']*?)["\'] button',
            r'button[^"^\']*["\']([^"^\']*?)["\']',
            r'button labeled ([^.!?]+)',
            r'click ([^.!?]+) button'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1).strip()
        
        return ""

    def _extract_field_name(self, text: str) -> str:
        """Extract field name from description"""
        patterns = [
            r'([a-zA-Z]+) field',
            r'enter ([a-zA-Z]+)',
            r'fill ([a-zA-Z]+)',
            r'input ([a-zA-Z]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1).strip()
        
        return ""

    def _extract_keyword_context(self, content: str, keyword: str, context_words: int = 20) -> List[str]:
        """Extract context around keywords"""
        contexts = []
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        
        for match in pattern.finditer(content):
            start = max(0, match.start() - context_words * 6)
            end = min(len(content), match.end() + context_words * 6)
            context = content[start:end]
            contexts.append(context.strip())
        
        return contexts

    def _extract_evidence_snippets(self, content: str) -> List[Dict[str, Any]]:
        """Extract evidence snippets for grounding test cases"""
        snippets = []
        
        # Look for key sentences that indicate functionality
        key_phrases = [
            'should be able to', 'must be able to', 'user can', 
            'click the', 'enter the', 'select the', 'navigate to',
            'verify that', 'check if', 'ensure that'
        ]
        
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            sentence = sentence.strip()
            if any(phrase in sentence.lower() for phrase in key_phrases) and len(sentence) > 20:
                snippets.append({
                    "text": sentence,
                    "type": "functional_requirement",
                    "confidence": 0.8,
                    "keywords": [phrase for phrase in key_phrases if phrase in sentence.lower()]
                })
        
        return snippets

    def _add_to_semantic_index(self, chunk: Dict[str, Any], doc_type: str):
        """Add chunk to semantic index"""
        self.text_chunks.append(chunk)
        self.semantic_index[chunk['chunk_id']] = {
            'doc_type': doc_type,
            'text': chunk['text'],
            'embedding': chunk['embedding']
        }

    def get_available_features(self) -> List[str]:
        """Get all detected features with confidence scores"""
        features = []
        for doc_type, doc_data in self.documents.items():
            features.extend(doc_data.get("features", []))
        
        # Sort by confidence
        features.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return features

    def get_document_summary(self) -> Dict[str, Any]:
        """Get comprehensive document summary"""
        summary = {
            "total_documents": len(self.documents),
            "available_features": self.get_available_features(),
            "total_chunks": len(self.text_chunks),
            "total_workflows": sum(len(doc.get("workflows", [])) for doc in self.documents.values()),
            "total_ui_elements": sum(len(doc.get("ui_elements", [])) for doc in self.documents.values()),
            "documents": {}
        }
        
        for doc_type, doc_data in self.documents.items():
            summary["documents"][doc_type] = {
                "features": doc_data.get("features", []),
                "workflows_count": len(doc_data.get("workflows", [])),
                "ui_elements_count": len(doc_data.get("ui_elements", [])),
                "validation_rules_count": len(doc_data.get("validation_rules", [])),
                "evidence_snippets_count": len(doc_data.get("evidence_snippets", []))
            }
        
        return summary

    def _extract_attributes(self, html: str) -> Dict[str, str]:
        """Extract attributes from HTML element"""
        attributes = {}
        attr_pattern = r'(\w+)=["\']([^"^\']*)["\']'
        matches = re.findall(attr_pattern, html)
        for key, value in matches:
            attributes[key] = value
        return attributes

    def _generate_selectors_from_html(self, html: str, element_type: str) -> List[str]:
        """Generate selectors from HTML element"""
        selectors = []
        attributes = self._extract_attributes(html)
        
        if 'id' in attributes:
            selectors.append(f'#{attributes["id"]}')
        
        if 'class' in attributes:
            classes = attributes['class'].split()
            for cls in classes:
                selectors.append(f'.{cls}')
        
        if 'name' in attributes:
            selectors.append(f'[name="{attributes["name"]}"]')
        
        # Type-based selectors
        if element_type == 'input' and 'type' in attributes:
            selectors.append(f'input[type="{attributes["type"]}"]')
        
        return selectors

    def _extract_html_context(self, content: str, html: str) -> str:
        """Extract context around HTML element"""
        start = content.find(html)
        if start == -1:
            return ""
        
        context_start = max(0, start - 100)
        context_end = min(len(content), start + len(html) + 100)
        return content[context_start:context_end]

    def _extract_field_from_context(self, text: str) -> str:
        """Extract field name from validation context"""
        patterns = [
            r'([a-zA-Z]+)\s+(?:field|input)',
            r'for\s+([a-zA-Z]+)',
            r'([a-zA-Z]+)\s+is required'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1).strip()
        
        return ""

    def _classify_workflow(self, workflow_text: str) -> str:
        """Classify workflow type"""
        text_lower = workflow_text.lower()
        for workflow_type, keywords in self.workflow_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return workflow_type
        return "general"

# Initialize enhanced knowledge base
knowledge_base = RAGEnhancedKnowledgeBase()
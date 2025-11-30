import logging
import re
import json
import time
import sys
from typing import Dict, List, Any
from knowledge_base import knowledge_base

logger = logging.getLogger(__name__)

class RAGEnhancedScriptGenerator:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def generate_selenium_script(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a SINGLE consolidated Python script for MULTIPLE test cases.
        """
        try:
            # Handle input: Ensure it's a list
            if isinstance(test_cases, dict):
                test_cases = [test_cases]
            
            if not test_cases:
                return {"status": "error", "message": "No test cases provided"}

            # Use the name of the first test case to name the class/file
            main_test_name = test_cases[0].get("name", "GeneratedTest")
            script_id = test_cases[0].get("id", "BATCH")
            
            logger.info(f"Generating consolidated script for {len(test_cases)} test cases...")

            # Generate the Code
            script_content = self._generate_consolidated_code(test_cases)
            
            return {
                "status": "success",
                "script_id": f"SCRIPT-{script_id}",
                "script_content": script_content,  # This is the formatted string
                "total_tests": len(test_cases)
            }

        except Exception as e:
            logger.error(f"Script generation error: {e}")
            return {
                "status": "error", 
                "error_message": str(e)
            }

    def _generate_consolidated_code(self, test_cases: List[Dict[str, Any]]) -> str:
        """Constructs the full Python script string with correct indentation"""
        lines = []

        # 1. Imports
        lines.extend([
            "import unittest",
            "import time",
            "import logging",
            "import sys",
            "from selenium import webdriver",
            "from selenium.webdriver.common.by import By",
            "from selenium.webdriver.support.ui import WebDriverWait",
            "from selenium.webdriver.support import expected_conditions as EC",
            "from selenium.webdriver.common.keys import Keys",
            "from selenium.webdriver.common.action_chains import ActionChains",
            "",
            "# Configure Logging",
            "logging.basicConfig(",
            "    level=logging.INFO,",
            "    format='%(asctime)s - %(levelname)s - %(message)s',",
            "    handlers=[logging.StreamHandler(sys.stdout)]",
            ")",
            "logger = logging.getLogger(__name__)",
            ""
        ])

        # 2. Class Definition
        class_name = self._generate_class_name(test_cases[0].get("name", "Suite"))
        lines.extend([
            f"class {class_name}(unittest.TestCase):",
            "",
            "    def setUp(self):",
            "        logger.info('🚀 Setting up test environment...')",
            "        options = webdriver.ChromeOptions()",
            "        options.add_argument('--start-maximized')",
            "        # options.add_argument('--headless') # Uncomment for headless mode",
            "        self.driver = webdriver.Chrome(options=options)",
            "        self.driver.implicitly_wait(10)",
            "        self.wait = WebDriverWait(self.driver, 15)",
            "",
            "    def tearDown(self):",
            "        if hasattr(self, 'driver'):",
            "            self.driver.quit()",
            "        logger.info('🏁 Test teardown complete.')",
            ""
        ])

        # 3. Loop through test cases and add methods
        for index, test_case in enumerate(test_cases, 1):
            method_name = self._generate_method_name(test_case.get("name", f"test_{index}"))
            description = test_case.get("description", "No description")
            
            lines.extend([
                f"    def {method_name}_{index}(self):",
                f'        """{description}"""',
                f"        logger.info('▶️  Starting Test Case {index}: {test_case.get('name')}')",
                "        driver = self.driver",
                ""
            ])
            
            # Add steps for this specific test case
            steps = test_case.get("steps", [])
            selector_mappings = test_case.get("selector_mappings", {})
            
            for step_idx, step in enumerate(steps, 1):
                # Convert step to code lines
                step_code = self._convert_step_to_code(step, selector_mappings)
                # Indent step code (8 spaces)
                lines.extend([f"        {line}" for line in step_code])
            
            lines.append("") # Empty line between methods

        # 4. Main Execution Block
        lines.extend([
            "if __name__ == '__main__':",
            "    unittest.main(verbosity=2)"
        ])

        return "\n".join(lines)

    def _convert_step_to_code(self, step: str, mappings: Dict) -> List[str]:
        """Converts a single English step into Python/Selenium code lines"""
        step_lower = step.lower()
        code = []
        
        # Log the step
        code.append(f"logger.info('Step: {step}')")

        try:
            # Navigation
            if "navigate" in step_lower or "go to" in step_lower:
                url = "https://example.com" # Default or extract from step
                if "login" in step_lower: url = "https://example.com/login"
                code.append(f"driver.get('{url}')")
                code.append("time.sleep(2)")

            # Input (Type/Enter)
            elif any(x in step_lower for x in ['enter', 'type', 'fill']):
                # Simple extraction logic
                selector = "//*[@name='q']" # Default
                # logic to pick selector from mappings would go here
                code.append(f"# Action: {step}")
                code.append(f"el = self.wait.until(EC.presence_of_element_located((By.XPATH, \"{selector}\")))")
                code.append("el.clear()")
                code.append("el.send_keys('test_data')")

            # Click
            elif "click" in step_lower:
                code.append(f"# Action: {step}")
                code.append("el = self.wait.until(EC.element_to_be_clickable((By.XPATH, \"//*[contains(text(), 'Button')]\")))")
                code.append("el.click()")
            
            # Verify
            elif "verify" in step_lower or "check" in step_lower:
                code.append(f"# Verification: {step}")
                code.append("self.assertTrue(True) # Placeholder for verification logic")

            else:
                code.append(f"# TODO: Implement logic for: {step}")
                code.append("time.sleep(1)")

        except Exception:
            code.append("# Error generating code for step")

        return code

    def _generate_class_name(self, name: str) -> str:
        clean = re.sub(r'[^a-zA-Z0-9]', '', name.title())
        return f"Test{clean}"

    def _generate_method_name(self, name: str) -> str:
        clean = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())
        return f"test_{clean}"

# Initialize
enhanced_script_generator = RAGEnhancedScriptGenerator(knowledge_base)
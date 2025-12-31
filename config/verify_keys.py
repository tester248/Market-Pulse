"""
API Key Verification and Management Utility
Verifies all configured API keys and tests basic functionality
"""

import os
import yaml
import requests
import asyncio
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

class APIKeyVerifier:
    """Verifies and tests API keys for all configured services"""
    
    def __init__(self):
        self.results = {}
        self.config_path = os.path.join(os.path.dirname(__file__), 'api_keys.yaml')
        
    def load_config(self) -> Dict:
        """Load API key configuration"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"❌ Configuration file not found: {self.config_path}")
            return {}
    
    def check_openai(self) -> Tuple[bool, str]:
        """Verify OpenAI API key"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return False, "API key not found in environment"
        
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            # Test with a simple completion
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1
            )
            return True, f"✅ Connected - Model: {response.model}"
        except Exception as e:
            return False, f"❌ Error: {str(e)}"
    
    def check_anthropic(self) -> Tuple[bool, str]:
        """Verify Anthropic API key"""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            return False, "API key not found in environment"
        
        try:
            headers = {
                'x-api-key': api_key,
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01'
            }
            
            data = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "Hello"}]
            }
            
            response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                return True, "✅ Connected - Claude API working"
            else:
                return False, f"❌ HTTP {response.status_code}: {response.text}"
        except Exception as e:
            return False, f"❌ Error: {str(e)}"
    
    def check_alpha_vantage(self) -> Tuple[bool, str]:
        """Verify Alpha Vantage API key"""
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            return False, "API key not found in environment"
        
        try:
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=1min&apikey={api_key}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if "Error Message" in data:
                return False, f"❌ {data['Error Message']}"
            elif "Note" in data:
                return False, f"❌ Rate limit: {data['Note']}"
            elif "Time Series (1min)" in data:
                return True, "✅ Connected - Stock data accessible"
            else:
                return False, f"❌ Unexpected response: {data}"
        except Exception as e:
            return False, f"❌ Error: {str(e)}"
    
    def check_polygon(self) -> Tuple[bool, str]:
        """Verify Polygon.io API key"""
        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key:
            return False, "API key not found in environment"
        
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?adjusted=true&apikey={api_key}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if response.status_code == 200 and data.get('status') == 'OK':
                return True, f"✅ Connected - {data.get('resultsCount', 0)} results"
            else:
                return False, f"❌ {data.get('error', 'Unknown error')}"
        except Exception as e:
            return False, f"❌ Error: {str(e)}"
    
    def check_coingecko(self) -> Tuple[bool, str]:
        """Verify CoinGecko API key"""
        api_key = os.getenv('COINGECKO_API_KEY')
        
        try:
            headers = {}
            if api_key:
                headers['x-cg-demo-api-key'] = api_key
            
            url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
            response = requests.get(url, headers=headers, timeout=10)
            data = response.json()
            
            if 'bitcoin' in data:
                price = data['bitcoin']['usd']
                key_status = "with API key" if api_key else "without API key (limited)"
                return True, f"✅ Connected {key_status} - BTC: ${price:,.2f}"
            else:
                return False, f"❌ Unexpected response: {data}"
        except Exception as e:
            return False, f"❌ Error: {str(e)}"
    
    def check_etherscan(self) -> Tuple[bool, str]:
        """Verify Etherscan API key"""
        api_key = os.getenv('ETHERSCAN_API_KEY')
        if not api_key:
            return False, "API key not found in environment"
        
        try:
            url = f"https://api.etherscan.io/api?module=stats&action=ethprice&apikey={api_key}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data.get('status') == '1':
                eth_price = data['result']['ethusd']
                return True, f"✅ Connected - ETH: ${eth_price}"
            else:
                return False, f"❌ {data.get('message', 'Unknown error')}"
        except Exception as e:
            return False, f"❌ Error: {str(e)}"
    
    def check_all_keys(self) -> Dict[str, Tuple[bool, str]]:
        """Check all configured API keys"""
        checks = {
            'OpenAI': self.check_openai,
            'Anthropic': self.check_anthropic,
            'Alpha Vantage': self.check_alpha_vantage,
            'Polygon.io': self.check_polygon,
            'CoinGecko': self.check_coingecko,
            'Etherscan': self.check_etherscan,
        }
        
        results = {}
        print("🔍 Verifying API Keys...\n")
        
        for service, check_func in checks.items():
            print(f"Checking {service}...", end=" ")
            try:
                success, message = check_func()
                results[service] = (success, message)
                print(message)
            except Exception as e:
                results[service] = (False, f"❌ Error: {str(e)}")
                print(results[service][1])
        
        return results
    
    def generate_report(self, results: Dict[str, Tuple[bool, str]]) -> None:
        """Generate a detailed verification report"""
        print(f"\n{'='*60}")
        print(f"API KEY VERIFICATION REPORT")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        working = []
        failed = []
        
        for service, (success, message) in results.items():
            if success:
                working.append(f"{service}: {message}")
            else:
                failed.append(f"{service}: {message}")
        
        if working:
            print(f"\n✅ WORKING APIs ({len(working)}):")
            for item in working:
                print(f"  • {item}")
        
        if failed:
            print(f"\n❌ FAILED APIs ({len(failed)}):")
            for item in failed:
                print(f"  • {item}")
            
            print("\n💡 TROUBLESHOOTING TIPS:")
            print("  1. Check .env file contains correct API keys")
            print("  2. Verify API keys are active and not expired")
            print("  3. Ensure sufficient credits/quota remaining")
            print("  4. Check network connectivity")
            print("  5. Review service-specific documentation")
        
        success_rate = len(working) / len(results) * 100
        print(f"\n📊 SUCCESS RATE: {success_rate:.1f}% ({len(working)}/{len(results)})")
        
        if success_rate == 100:
            print("🎉 All APIs are working perfectly!")
        elif success_rate >= 80:
            print("✅ Most APIs are working well!")
        elif success_rate >= 50:
            print("⚠️  Some APIs need attention")
        else:
            print("🚨 Multiple API issues detected")
        
        print(f"\n📚 For help setting up API keys, see: config/API_KEYS_GUIDE.md")
        print(f"{'='*60}")

def main():
    """Main verification function"""
    verifier = APIKeyVerifier()
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("📋 Please copy .env.template to .env and add your API keys")
        print("   cp .env.template .env")
        return
    
    # Load and check environment variables
    load_dotenv()
    
    # Run verification
    results = verifier.check_all_keys()
    verifier.generate_report(results)

if __name__ == "__main__":
    main()
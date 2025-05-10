import requests
import face_recognition
import cv2
import os
import re
import time
import threading
import concurrent.futures
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from urllib.parse import quote_plus, urlparse
from fake_useragent import UserAgent
from collections import defaultdict

class PersonInfoScanner:
    def __init__(self):
        self.ua = UserAgent()
        self.lock = threading.Lock()
        self.results = defaultdict(set)
        self.face_encodings = []
        self.platforms = {
            'google': self._search_google,
            'duckduckgo': self._search_duckduckgo
        }
        self.possible_info = {
            'names': set(),
            'gender': None,
            'emails': set(),
            'phones': set(),
            'birth_info': set(),
            'nationality': set(),
            'social_media': defaultdict(set)
        }
        
    def _get_random_headers(self):
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def _load_face(self, image_path):
        try:
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                print(f"No faces detected in {image_path}")
                return False
            
            self.face_encodings = [face_recognition.face_encodings(image, [location])[0] 
                                   for location in face_locations]
            print(f"Loaded {len(self.face_encodings)} face(s) from image")
            return True
        except Exception as e:
            print(f"Error loading face image: {e}")
            return False
    
    def _search_google(self, query, is_face_search=False):
        search_url = f"https://www.google.com/search?q={quote_plus(query)}"
        if is_face_search:
            search_url += "&tbm=isch"
        
        try:
            response = requests.get(search_url, headers=self._get_random_headers(), timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                if is_face_search:
                    img_elements = soup.find_all('img')
                    for img in img_elements[1:]:  # Skip Google logo
                        if img.get('src') and img['src'].startswith('http'):
                            try:
                                img_response = requests.get(img['src'], headers=self._get_random_headers(), timeout=5)
                                if img_response.status_code == 200:
                                    img_data = BytesIO(img_response.content)
                                    unknown_image = face_recognition.load_image_file(img_data)
                                    unknown_face_locations = face_recognition.face_locations(unknown_image)
                                    
                                    if unknown_face_locations:
                                        unknown_encodings = face_recognition.face_encodings(unknown_image, unknown_face_locations)
                                        for unknown_encoding in unknown_encodings:
                                            matches = face_recognition.compare_faces(self.face_encodings, unknown_encoding, tolerance=0.6)
                                            if any(matches):
                                                parent = img.parent
                                                if parent:
                                                    link = parent.get('href')
                                                    if link:
                                                        self._extract_info_from_url(link)
                            except Exception as e:
                                continue
                else:
                    links = soup.find_all('a')
                    for link in links:
                        href = link.get('href')
                        if href and href.startswith('/url?q='):
                            url = href.split('/url?q=')[1].split('&')[0]
                            self._extract_info_from_url(url)
                            
                # Extract text data
                text_content = soup.get_text()
                self._extract_patterns_from_text(text_content)
            
        except Exception as e:
            print(f"Error in Google search: {e}")
    
    def _search_duckduckgo(self, query, is_face_search=False):
        search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        if is_face_search:
            search_url += " images"
        
        try:
            response = requests.get(search_url, headers=self._get_random_headers(), timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                links = soup.find_all('a', {'class': 'result__url'})
                for link in links:
                    href = link.get('href')
                    if href:
                        self._extract_info_from_url(href)
                
                # Extract text data
                text_content = soup.get_text()
                self._extract_patterns_from_text(text_content)
                
                if is_face_search:
                    # For DuckDuckGo, we need to make a separate request for images
                    img_search_url = f"https://duckduckgo.com/i.js?q={quote_plus(query)}&o=json"
                    try:
                        img_response = requests.get(img_search_url, headers=self._get_random_headers(), timeout=10)
                        if img_response.status_code == 200 and 'json' in img_response.headers.get('content-type', ''):
                            data = img_response.json()
                            for result in data.get('results', []):
                                try:
                                    img_url = result.get('image')
                                    if img_url:
                                        img_response = requests.get(img_url, headers=self._get_random_headers(), timeout=5)
                                        if img_response.status_code == 200:
                                            img_data = BytesIO(img_response.content)
                                            unknown_image = face_recognition.load_image_file(img_data)
                                            unknown_face_locations = face_recognition.face_locations(unknown_image)
                                            
                                            if unknown_face_locations:
                                                unknown_encodings = face_recognition.face_encodings(unknown_image, unknown_face_locations)
                                                for unknown_encoding in unknown_encodings:
                                                    matches = face_recognition.compare_faces(self.face_encodings, unknown_encoding, tolerance=0.6)
                                                    if any(matches):
                                                        source_url = result.get('url')
                                                        if source_url:
                                                            self._extract_info_from_url(source_url)
                                except Exception as e:
                                    continue
                    except Exception as e:
                        print(f"Error in DuckDuckGo image search: {e}")
            
        except Exception as e:
            print(f"Error in DuckDuckGo search: {e}")
    
    def _extract_info_from_url(self, url):
        try:
            response = requests.get(url, headers=self._get_random_headers(), timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                text_content = soup.get_text()
                self._extract_patterns_from_text(text_content)
                
                # Check for social media links
                social_patterns = {
                    'facebook': r'facebook\.com/([^/"\']+)',
                    'twitter': r'twitter\.com/([^/"\']+)',
                    'instagram': r'instagram\.com/([^/"\']+)',
                    'linkedin': r'linkedin\.com/in/([^/"\']+)',
                    'github': r'github\.com/([^/"\']+)'
                }
                
                for platform, pattern in social_patterns.items():
                    matches = re.findall(pattern, response.text)
                    for match in matches:
                        if match and len(match) > 1 and not match.startswith(('img', 'static', 'assets', 'js', 'css')):
                            with self.lock:
                                self.possible_info['social_media'][platform].add(match)
        
        except Exception as e:
            pass
    
    def _extract_patterns_from_text(self, text):
        # Extract potential names (capitalized words next to each other)
        name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b'
        names = re.findall(name_pattern, text)
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        
        # Extract phone numbers (various formats)
        phone_pattern = r'\b(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?(?:\d{3}[-.\s]?\d{4})\b'
        phones = re.findall(phone_pattern, text)
        
        # Extract birth information
        birth_pattern = r'(?:born|birth|birthday)(?:[\s:]+)([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?[,\s]+\d{4}|\d{1,2}(?:st|nd|rd|th)?[,\s]+[A-Za-z]+[,\s]+\d{4}|\d{4})'
        birth_info = re.findall(birth_pattern, text, re.IGNORECASE)
        
        # Extract nationality/origin
        origin_pattern = r'(?:from|nationality|origin|born in)(?:[\s:]+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        origins = re.findall(origin_pattern, text, re.IGNORECASE)
        
        # Gender detection
        male_indicators = re.findall(r'\b(?:he|him|his|male|man|boy|gentleman|mr\.?|mister)\b', text.lower())
        female_indicators = re.findall(r'\b(?:she|her|hers|female|woman|girl|lady|ms\.?|mrs\.?|miss)\b', text.lower())
        
        with self.lock:
            for name in names:
                self.possible_info['names'].add(name)
            
            for email in emails:
                self.possible_info['emails'].add(email)
                
            for phone in phones:
                self.possible_info['phones'].add(phone)
                
            for info in birth_info:
                self.possible_info['birth_info'].add(info.strip())
                
            for origin in origins:
                self.possible_info['nationality'].add(origin.strip())
            
            # Simple gender determination
            if not self.possible_info['gender']:
                male_count = len(male_indicators)
                female_count = len(female_indicators)
                
                if male_count > female_count * 2:  # Ensure strong confidence
                    self.possible_info['gender'] = 'Male'
                elif female_count > male_count * 2:
                    self.possible_info['gender'] = 'Female'
    
    def worker(self, platform, query, is_face_search=False):
        search_function = self.platforms.get(platform)
        if search_function:
            search_function(query, is_face_search)
    
    def search(self, name, face_path):
        if not self._load_face(face_path):
            return "Failed to load face image. Please ensure the image contains a clearly visible face."
        
        queries = [
            name,
            f"{name} profile",
            f"{name} contact",
            f"{name} about",
            f"{name} information",
            f"{name} social media"
        ]
        
        # Create thread pools
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit face search tasks (3 workers per platform)
            face_futures = []
            for platform in self.platforms:
                for _ in range(3):
                    face_futures.append(executor.submit(self.worker, platform, name, True))
            
            # Submit text search tasks (3 workers per platform)
            text_futures = []
            for platform in self.platforms:
                for query in queries:
                    text_futures.append(executor.submit(self.worker, platform, query, False))
            
            # Wait for all tasks to complete
            concurrent.futures.wait(face_futures + text_futures)
        
        # Compile results
        return self._compile_results()
    
    def _compile_results(self):
        results = {}
        
        # Verify name matches
        search_name_parts = self.input_name.lower().split()
        verified_names = []
        
        for name in self.possible_info['names']:
            name_parts = name.lower().split()
            matches = sum(p in name_parts for p in search_name_parts)
            if matches / max(len(search_name_parts), len(name_parts)) >= 0.5:  # At least 50% match
                verified_names.append(name)
        
        # Get most common name
        if verified_names:
            name_count = defaultdict(int)
            for name in verified_names:
                name_count[name] += 1
            most_common_name = max(name_count.items(), key=lambda x: x[1])[0]
            results["Real Name (verified)"] = most_common_name
        else:
            results["Real Name"] = "Unknown"
        
        # Add gender if found
        if self.possible_info['gender']:
            results["Gender"] = self.possible_info['gender']
        else:
            results["Gender"] = "Unknown"
        
        # Add emails if found
        if self.possible_info['emails']:
            results["Email(s)"] = list(self.possible_info['emails'])
        else:
            results["Email"] = "Not found"
        
        # Add phones if found
        if self.possible_info['phones']:
            results["Phone Number(s)"] = list(self.possible_info['phones'])
        else:
            results["Phone"] = "Not found"
        
        # Add birth info if found
        if self.possible_info['birth_info']:
            results["Birth Information"] = list(self.possible_info['birth_info'])
        else:
            results["Birth Information"] = "Not found"
        
        # Add nationality if found
        if self.possible_info['nationality']:
            results["Nationality/Origin"] = list(self.possible_info['nationality'])
        else:
            results["Nationality/Origin"] = "Not found"
        
        # Add social media
        if self.possible_info['social_media']:
            social_results = {}
            for platform, usernames in self.possible_info['social_media'].items():
                if usernames:
                    social_results[platform] = list(usernames)
            
            if social_results:
                results["Social Media"] = social_results
            else:
                results["Social Media"] = "Not found"
        else:
            results["Social Media"] = "Not found"
        
        return results

    def run(self):
        self.input_name = input("Enter person's full name (all lowercase): ").strip()
        face_path = input("Enter path to face image (in current folder): ").strip()
        
        # Make sure we're using a path in the current directory
        if os.path.dirname(face_path):
            print("Please provide just the filename in the current directory.")
            return
        
        if not os.path.exists(face_path):
            print(f"Error: Image file '{face_path}' not found in current directory.")
            return
        
        print(f"\nScanning for information about '{self.input_name}'...")
        print(f"Using face image: {face_path}")
        print("\nThis may take a few minutes. Scanning multiple sources...\n")
        
        start_time = time.time()
        results = self.search(self.input_name, face_path)
        end_time = time.time()
        
        print(f"\nScan completed in {end_time - start_time:.2f} seconds.\n")
        
        if isinstance(results, str):
            print(results)  # Error message
            return
        
        print("=" * 50)
        print("INFORMATION FOUND:")
        print("=" * 50)
        
        for key, value in results.items():
            if isinstance(value, list):
                print(f"{key}:")
                for item in value[:5]:  # Limit to top 5 results per category
                    print(f"  - {item}")
            elif isinstance(value, dict):
                print(f"{key}:")
                for platform, usernames in value.items():
                    print(f"  {platform.capitalize()}:")
                    for username in list(usernames)[:3]:  # Limit to top 3 results per platform
                        print(f"    - {username}")
            else:
                print(f"{key}: {value}")
        
        print("\nNote: Some information may be inaccurate. Verify results independently.")
        return results

if __name__ == "__main__":
    scanner = PersonInfoScanner()
    scanner.run()

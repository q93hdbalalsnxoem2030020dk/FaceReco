import os
import requests
import bs4
import cv2
import PIL.Image
import concurrent.futures
import urllib.parse
import re
import io
import numpy as np
import face_recognition
import time
import hashlib
import random
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

class FaceAnalyzer:
    def __init__(self):
        self.face_detector_models = {
            'haar': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
            'lbp': cv2.CascadeClassifier(cv2.data.haarcascades + 'lbpcascade_frontalface.xml')
        }
        # Parameters for face detection tuning
        self.detection_params = {
            'scale_factor': 1.05,  # Smaller scale factor for better detection
            'min_neighbors': 3,    # Fewer neighbors for more aggressive detection
            'jitter': 5,           # Number of times to sample face for encoding
            'model': 'large'       # Use CNN model for better accuracy when available
        }
    
    def analyze_face(self, image_path):
        """Enhanced face analysis with multiple detection methods and better parameters"""
        try:
            # Try to open and process the image
            img_pil = PIL.Image.open(image_path).convert('RGB')
            img_np = np.array(img_pil)
            
            # Use face_recognition (HOG-based) first
            face_locations = face_recognition.face_locations(
                img_np, 
                model="cnn" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "hog"  # Use CNN if CUDA available
            )
            
            # If no faces, try with different parameters
            if not face_locations:
                # Try with different parameters (more aggressive)
                face_locations = face_recognition.face_locations(
                    img_np,
                    number_of_times_to_upsample=2  # Upsample for better detection of small faces
                )
            
            face_recognition_success = len(face_locations) > 0
            
            # Try OpenCV methods as fallback and validation
            opencv_results = {}
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img_cv_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply image enhancements to improve detection
            img_cv_equalized = cv2.equalizeHist(img_cv_gray)
            
            # Try multiple detectors with different parameters
            for name, detector in self.face_detector_models.items():
                faces = detector.detectMultiScale(
                    img_cv_equalized,
                    scaleFactor=self.detection_params['scale_factor'],
                    minNeighbors=self.detection_params['min_neighbors'],
                    minSize=(30, 30)
                )
                opencv_results[name] = len(faces) > 0
            
            # If no face was found with any method, return failure
            if not face_recognition_success and not any(opencv_results.values()):
                return False, "No face detected in the image by any detection method"
            
            # If face_recognition detected faces, encode them
            face_encoding = None
            if face_recognition_success:
                # Get face encoding with multiple samples for better accuracy
                face_encoding = face_recognition.face_encodings(
                    img_np, 
                    face_locations, 
                    num_jitters=self.detection_params['jitter'],
                    model=self.detection_params['model']
                )[0]
            
            return True, {
                "face_detected_fr": face_recognition_success,
                "face_encoding": face_encoding,
                "face_location_fr": face_locations[0] if face_recognition_success else None,
                "opencv_detections": opencv_results
            }
        
        except Exception as e:
            return False, f"Face analysis error: {str(e)}"

    def compare_face_encodings(self, known_encoding, unknown_encoding, tolerance=0.5):
        """Compare face encodings with enhanced accuracy and metrics"""
        if known_encoding is None or unknown_encoding is None:
            return False, 1.0
            
        # Calculate Euclidean distance
        distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
        
        # Advanced matching logic - use stricter tolerance for better accuracy
        return distance <= tolerance, distance
    
    def download_and_analyze_image(self, image_url, session):
        """Download and analyze an image with error handling"""
        try:
            # Use the session for connection pooling and retry handling
            response = session.get(image_url, timeout=5, stream=True)
            response.raise_for_status()
            
            # Check if it's actually an image by content type
            if 'image' not in response.headers.get('Content-Type', ''):
                return None, None
                
            img_bytes = response.content
            img_io = io.BytesIO(img_bytes)
            img_pil = PIL.Image.open(img_io).convert('RGB')
            img_np = np.array(img_pil)
            
            # Use CNN model if CUDA is available
            detection_model = "cnn" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "hog"
            
            # Detect faces with different parameters for better results
            face_locations = face_recognition.face_locations(
                img_np, 
                model=detection_model, 
                number_of_times_to_upsample=1
            )
            
            if not face_locations:
                return None, None
                
            # Enhanced encoding with multiple samples
            face_encoding = face_recognition.face_encodings(
                img_np, 
                face_locations, 
                num_jitters=self.detection_params['jitter'],
                model=self.detection_params['model']
            )[0]
                
            return face_locations[0], face_encoding
            
        except (requests.exceptions.RequestException, PIL.UnidentifiedImageError):
            return None, None
        except Exception:
            return None, None


class OSINTSearcher:
    def __init__(self):
        self.search_engines = ['google', 'duckduckgo', 'bing', 'yandex']
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:94.0) Gecko/20100101 Firefox/94.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/96.0.1054.62'
        ]
        self.retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504]
        )
    
    def create_session(self):
        """Create a requests session with retry logic and random user agent"""
        session = requests.Session()
        session.mount('https://', HTTPAdapter(max_retries=self.retries))
        session.mount('http://', HTTPAdapter(max_retries=self.retries))
        session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Referer': 'https://www.google.com/'
        })
        return session
    
    def get_search_url(self, platform, query, page=0):
        """Get URL for different search engines with pagination"""
        encoded_query = urllib.parse.quote(query)
        
        if platform == 'google':
            return f'https://www.google.com/search?q={encoded_query}&start={page*10}'
        elif platform == 'duckduckgo':
            return f'https://duckduckgo.com/html/?q={encoded_query}'
        elif platform == 'bing':
            return f'https://www.bing.com/search?q={encoded_query}&first={page*10}'
        elif platform == 'yandex':
            return f'https://yandex.com/search/?text={encoded_query}&p={page}'
        return None

    def find_image_urls_in_soup(self, soup, base_url):
        """Find and extract all image URLs from a webpage"""
        image_urls = []
        
        # Standard image tags
        for img_tag in soup.find_all('img'):
            src = img_tag.get('src') or img_tag.get('data-src') or img_tag.get('data-original')
            if src:
                full_url = urllib.parse.urljoin(base_url, src)
                image_urls.append(full_url)
        
        # Background images in style attributes
        for tag in soup.find_all(lambda tag: tag.has_attr('style') and 'background' in tag['style']):
            style = tag['style']
            urls = re.findall(r'url\(["\']?([^"\'()]+)["\']?\)', style)
            for url in urls:
                full_url = urllib.parse.urljoin(base_url, url)
                image_urls.append(full_url)
        
        # Meta tags with images
        for meta in soup.find_all('meta'):
            content = meta.get('content', '')
            if content and content.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                full_url = urllib.parse.urljoin(base_url, content)
                image_urls.append(full_url)
        
        return image_urls
        
    def perform_search(self, platform, name, session):
        """Perform search on a platform with advanced search queries"""
        results = []
        image_urls_found = set()  # Use set to avoid duplicates
        
        # Generate search queries with variations and specific information markers
        queries = [
            f'"{name}" personal information contact biography',
            f'"{name}" profile social media',
            f'"{name}" email phone contact "about me"',
            f'"{name}" date of birth location',
            f'"{name}" personal photos profile pictures'
        ]
        
        # Try multiple search queries for more comprehensive results
        for query in queries[:2]:  # Limit to first 2 queries for efficiency
            url = self.get_search_url(platform, query)
            if not url:
                continue
                
            try:
                response = session.get(url, timeout=10)
                if response.status_code != 200:
                    continue
                    
                soup = bs4.BeautifulSoup(response.text, 'html.parser')
                
                # Extract results based on search engine
                if platform == 'google':
                    base_url = 'https://www.google.com/'
                    # Google search result selectors (updated for 2024)
                    for result_div in soup.select('div.g, div.xpd, div[data-hveid]'):
                        link_tag = result_div.find('a')
                        if not link_tag or not link_tag.get('href', '').startswith('http'):
                            continue
                            
                        link = link_tag['href']
                        title_tag = result_div.find('h3')
                        title = title_tag.text if title_tag else 'N/A'
                        snippet_tag = result_div.select_one('div.VwiC3b, div.lEBKkf, div.BNeawe')
                        snippet = snippet_tag.text if snippet_tag else 'No snippet available.'
                        
                        # Only process results that look relevant
                        if self._is_relevant_result(title, snippet, name):
                            results.append({
                                'source': 'Google',
                                'title': title,
                                'link': link,
                                'snippet': snippet
                            })
                            
                            # Check linked page for more information and images
                            self._process_linked_page(link, session, image_urls_found, results)
                
                elif platform == 'duckduckgo':
                    base_url = 'https://duckduckgo.com/'
                    for result in soup.select('.result, .web-result'):
                        title_tag = result.select_one('.result__a, .result__title a')
                        if not title_tag:
                            continue
                            
                        title = title_tag.text
                        link = title_tag.get('href')
                        if not link:
                            continue
                            
                        snippet_tag = result.select_one('.result__snippet')
                        snippet = snippet_tag.text if snippet_tag else 'No snippet available.'
                        
                        if self._is_relevant_result(title, snippet, name):
                            results.append({
                                'source': 'DuckDuckGo',
                                'title': title,
                                'link': link,
                                'snippet': snippet
                            })
                            
                            self._process_linked_page(link, session, image_urls_found, results)
                
                elif platform == 'bing':
                    base_url = 'https://www.bing.com/'
                    for result in soup.select('li.b_algo'):
                        title_tag = result.select_one('h2 a')
                        if not title_tag:
                            continue
                            
                        title = title_tag.text
                        link = title_tag.get('href')
                        if not link:
                            continue
                            
                        snippet_tag = result.select_one('.b_caption p')
                        snippet = snippet_tag.text if snippet_tag else 'No snippet available.'
                        
                        if self._is_relevant_result(title, snippet, name):
                            results.append({
                                'source': 'Bing',
                                'title': title,
                                'link': link,
                                'snippet': snippet
                            })
                            
                            self._process_linked_page(link, session, image_urls_found, results)
                
                elif platform == 'yandex':
                    base_url = 'https://yandex.com/'
                    for result in soup.select('li.serp-item'):
                        title_tag = result.select_one('h2 a')
                        if not title_tag:
                            continue
                            
                        title = title_tag.text
                        link = title_tag.get('href')
                        if not link:
                            continue
                            
                        snippet_tag = result.select_one('.text-container')
                        snippet = snippet_tag.text if snippet_tag else 'No snippet available.'
                        
                        if self._is_relevant_result(title, snippet, name):
                            results.append({
                                'source': 'Yandex',
                                'title': title,
                                'link': link,
                                'snippet': snippet
                            })
                            
                            self._process_linked_page(link, session, image_urls_found, results)
                            
                # Look for image URLs in the search results page itself
                page_image_urls = self.find_image_urls_in_soup(soup, base_url)
                image_urls_found.update(page_image_urls)
                
            except Exception as e:
                continue
        
        return results, list(image_urls_found)
    
    def _is_relevant_result(self, title, snippet, name):
        """Check if search result looks relevant to the person"""
        name_parts = name.lower().split()
        
        # Check if all parts of the name appear in title or snippet
        name_match = all(part in title.lower() or part in snippet.lower() for part in name_parts)
        
        # Look for indicators of personal information
        personal_info_indicators = ['profile', 'bio', 'about', 'contact', 'social', 'media', 
                                   'facebook', 'twitter', 'linkedin', 'instagram', 'email']
                                   
        has_indicators = any(indicator in (title + snippet).lower() for indicator in personal_info_indicators)
        
        return name_match and has_indicators
    
    def _process_linked_page(self, link, session, image_urls_found, results):
        """Process a linked page to extract more information"""
        try:
            # Skip certain file types and domains
            if re.search(r'\.(pdf|doc|docx|ppt|pptx|xls|xlsx)$', link, re.I):
                return
                
            # Skip certain domains that are unlikely to have personal info
            if any(domain in link for domain in ['youtube.com/watch', 'amazon.com/product']):
                return
                
            page_response = session.get(link, timeout=5, allow_redirects=True)
            if page_response.status_code != 200:
                return
                
            # Skip non-HTML content
            content_type = page_response.headers.get('Content-Type', '')
            if 'text/html' not in content_type:
                return
                
            page_soup = bs4.BeautifulSoup(page_response.text, 'html.parser')
            
            # Get images from the linked page
            page_images = self.find_image_urls_in_soup(page_soup, link)
            image_urls_found.update(page_images)
            
            # Extract and store full text (limited to reduce size)
            for i, result in enumerate(results):
                if result.get('link') == link:
                    # Get visible text without script/style content
                    for script in page_soup(["script", "style"]):
                        script.extract()
                    
                    text = page_soup.get_text(separator=' ', strip=True)
                    results[i]['full_text_snippet'] = text[:3000]  # Increase limit for better info extraction
                    break
        
        except Exception:
            pass


class InfoExtractor:
    def __init__(self):
        # Enhanced pattern sets for information extraction
        self.patterns = {
            'emails': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phones': [
                r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
                r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
                r'\b\+\d{1,3}\s?\d{10,14}\b'
            ],
            'social_media': {
                'facebook': r'(?:https?://)?(?:www\.)?(?:facebook|fb)\.com/(?:(?:\w)*#!)?(?:pages/)?(?:[?\w\-]*/)?([\w\-\.]*)',
                'twitter': r'(?:https?://)?(?:www\.)?(?:twitter\.com|x\.com)/(?:#!/)?(?@)?([\w\-\.]*)',
                'linkedin': r'(?:https?://)?(?:[\w]+\.)?linkedin\.com/in/([A-Za-z0-9_-]+)',
                'instagram': r'(?:https?://)?(?:www\.)?instagram\.com/([A-Za-z0-9_.]+)',
                'github': r'(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_-]+)',
                'pinterest': r'(?:https?://)?(?:www\.)?pinterest\.com/([A-Za-z0-9_-]+)',
                'tumblr': r'(?:https?://)?(?:www\.)?([a-zA-Z0-9_-]+)\.tumblr\.com',
                'youtube': r'(?:https?://)?(?:www\.)?youtube\.com/(?:c/|channel/|user/)?([a-zA-Z0-9_-]+)',
                'reddit': r'(?:https?://)?(?:www\.)?reddit\.com/(?:u|user)/([A-Za-z0-9_-]+)',
                'tiktok': r'(?:https?://)?(?:www\.)?tiktok\.com/@([A-Za-z0-9_.]+)',
                'snapchat': r'(?:https?://)?(?:www\.)?snapchat\.com/add/([A-Za-z0-9_.]+)'
            },
            'birth_dates': [
                r'\b(?:born\s+on|birthdate|date\s+of\s+birth|birth\s+date|DOB)[\s:]+([A-Za-z0-9\s,.]+\d{4})\b',
                r'\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b',
                r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?),?\s+\d{4}\b',
                r'\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b'
            ],
            'locations': [
                r'\b(?:lives\s+in|based\s+in|located\s+in|resides\s+in|from|home\s+is\s+in|hometown)[\s:]+([A-Z][a-zA-Z\s,.]+?(?:,\s*[A-Z]{2})?)(?=[,.;]|\s+(?:is|and|with|since|where|who|but|for)\b|\s*$)',
                r'\b(?:born\s+in)[\s:]+([A-Z][a-zA-Z\s,.]+?(?:,\s*[A-Z]{2})?)(?=[,.;]|\s+(?:is|and|with|since|where|who|but|for)\b|\s*$)',
                r'(?<=\s)([A-Z][a-zA-Z]+(?:,\s*[A-Z]{2})?(?:,\s*[A-Z][a-zA-Z\s]+)?)(?:-based\b)'
            ],
            'employers': [
                r'\b(?:works\s+(?:at|for)|employed\s+(?:at|by)|job\s+at)[\s:]+([A-Z][a-zA-Z0-9\s&,.\'"-]+?)(?:[,.;]|\s+(?:as|in|since|from|where|and|with)\b|\s*$)',
                r'\b(?:employer|company)[\s:]+([A-Z][a-zA-Z0-9\s&,.\'"-]+?)(?:[,.;]|\s+(?:as|in|since|from|where|and|with)\b|\s*$)'
            ],
            'job_titles': [
                r'\b(?:is\s+an?|is\s+the|works\s+as\s+an?|works\s+as\s+the|employed\s+as\s+an?|employed\s+as\s+the|position)[\s:]+([A-Z][a-zA-Z\s&,.\'"-]+?)(?:[,.;]|\s+(?:at|in|since|from|where|and|with)\b|\s*$)',
                r'\b([A-Z][a-zA-Z]+\s+(?:Engineer|Director|Manager|Officer|Developer|Designer|Analyst|Consultant|Specialist|Executive|Administrator|Coordinator))(?:[,.;]|\s+(?:at|in|since|from|where|and|with)\b|\s*$)'
            ],
            'education': [
                r'\b(?:studied\s+at|attended|graduated\s+from|alumnus\s+of|alumni\s+of|education|degree\s+from|student\s+at)[\s:]+([A-Z][a-zA-Z0-9\s&,.\'"-]+?)(?:[,.;]|\s+(?:in|with|where|and|class)\b|\s*$)',
                r'\b([A-Z][a-zA-Z\s]+\s+(?:University|College|School|Institute|Academy))(?:[,.;]|\s+(?:in|with|where|and|class)\b|\s*$)'
            ],
            'skills': [
                r'\b(?:skills|expertise|proficient\s+in|specializes\s+in|expert\s+in)[\s:]+([^,.;]{3,100}?)(?:[,.;]|\s+(?:and|with|for|as)\b|\s*$)'
            ],
            'usernames': [
                r'@([A-Za-z][A-Za-z0-9_]{2,15})\b',
                r'\b(?:username|user|handle)[\s:]+([A-Za-z][A-Za-z0-9_.-]{2,20})\b'
            ],
            'interests': [
                r'\b(?:interests|hobbies|enjoys|passionate\s+about|fan\s+of)[\s:]+([^,.;]{3,100}?)(?:[,.;]|\s+(?:and|with|for|as)\b|\s*$)'
            ]
        }

    def extract_info_from_text(self, text):
        """Extract structured information from text using enhanced patterns"""
        extracted = {}
        
        # Process basic patterns
        for key, pattern in self.patterns.items():
            if key == 'social_media':
                extracted[key] = {}
                for platform, platform_pattern in pattern.items():
                    matches = re.findall(platform_pattern, text, re.IGNORECASE)
                    if matches:
                        extracted[key][platform] = list(set(matches))
            elif isinstance(pattern, list):
                extracted[key] = []
                for p in pattern:
                    matches = re.findall(p, text, re.IGNORECASE)
                    extracted[key].extend([m[0] if isinstance(m, tuple) else m for m in matches])
                extracted[key] = list(set(extracted[key]))
            else:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    extracted[key] = list(set(matches))
        
        # Clean extracted data
        self._clean_extracted_data(extracted)
        
        return extracted
    
    def _clean_extracted_data(self, data):
        """Clean up the extracted data to remove noise and false positives"""
        
        # Clean up phone numbers
        if 'phones' in data and data['phones']:
            cleaned_phones = []
            for phone in data['phones']:
                # Remove known false positives (like dates mistaken as phone numbers)
                if re.search(r'19[0-9]{2}|20[0-2][0-9]', phone):  # Skip years
                    continue
                # Format consistently
                phone = re.sub(r'[^\d+]', '', phone)  # Keep only digits and + sign
                if len(phone) >= 7:  # Must have at least 7 digits to be a valid phone
                    cleaned_phones.append(phone)
            data['phones'] = list(set(cleaned_phones))
        
        # Clean up locations
        if 'locations' in data and data['locations']:
            cleaned_locations = []
            for location in data['locations']:
                # Remove entries that are just single words (likely false positives)
                if ' ' in location:
                    # Remove trailing periods and other punctuation
                    location = re.sub(r'[.,;:]$', '', location).strip()
                    cleaned_locations.append(location)
            data['locations'] = list(set(cleaned_locations))
        
        # Clean up emails
        if 'emails' in data and data['emails']:
            cleaned_emails = []
            for email in data['emails']:
                # Verify email looks legitimate
                if '.' in email.split('@')[1]:  # Must have domain with a period
                    cleaned_emails.append(email.lower())
            data['emails'] = list(set(cleaned_emails))


class OSINTTool:
    def __init__(self):
        self.face_analyzer = FaceAnalyzer()
        self.searcher = OSINTSearcher()
        self.extractor = InfoExtractor()
        
    def run(self, person_name, image_path):
        print(f"OSINT and facial recognition analysis for: {person_name}")
        print("-" * 60)
        
        # Step 1: Face Analysis
        print("\nAnalyzing face image...")
        face_analysis_success, face_analysis_result = self.face_analyzer.analyze_face(image_path)
        
        if not face_analysis_success:
            print(f"Face analysis failed: {face_analysis_result}")
            face_data = None
            input_face_encoding = None
        else:
            print("✅ Face analysis successful")
            face_data = face_analysis_result
            input_face_encoding = face_data.get("face_encoding")
        
        # Step 2: Search for information
        print("\nInitiating concurrent search across multiple engines...")
        start_time = time.time()
        all_results = []
        all_image_urls = []
        
        # Create session for connection pooling and consistent headers
        session = self.searcher.create_session()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            
            # Submit search tasks for each engine
            for engine in ['google', 'duckduckgo', 'bing', 'yandex']:
                futures[executor.submit(self.searcher.perform_search, engine, person_name, session)] = engine
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                engine = futures[future]
                try:
                    results, image_urls = future.result()
                    all_results.extend(results)
                    all_image_urls.extend(image_urls)
                    print(f"  ✅ {engine.capitalize()} search completed: {len(results)} results, {len(image_urls)} images")
                except Exception as exc:
                    print(f"  ❌ {engine.capitalize()} search failed: {exc}")
        
        search_end_time = time.time()
        print(f"Search completed in {search_end_time - start_time:.2f} seconds. Found {len(all_results)} total results and {len(all_image_urls)} images.")
        
        # Step 3: Extract information from search results
        print("\nExtracting personal information from search results...")
        combined_text = ""
        
        # Combine all text from search results
        for result in all_results:
            if 'title' in result:
                combined_text += result['title'] + " "
            if 'snippet' in result:
                combined_text += result['snippet'] + " "
            if 'full_text_snippet' in result:
                combined_text += result['full_text_snippet'] + " "
        
        # Extract structured information
        extracted_info = self.extractor.extract_info_from_text(combined_text)
        
        # Clean up and prioritize information
        personal_info = self._cleanup_extracted_info(extracted_info, person_name)
        
        print("✅ Information extraction complete")
        
        # Step 4: Analyze images for face matching
        print("\nAnalyzing images for face matching...")
        face_match_results = []
        
        if input_face_encoding is not None:
            # Deduplicate image URLs
            unique_image_urls = list(set(all_image_urls))
            print(f"Processing {len(unique_image_urls)} unique images...")
            
            # Create image clusters for batch processing
            image_batches = [unique_image_urls[i:i+10] for i in range(0, len(unique_image_urls), 10)]
            
            face_matches_found = 0
            
            # Process images in parallel batches
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                for batch_idx, image_batch in enumerate(image_batches):
                    print(f"  Processing batch {batch_idx+1}/{len(image_batches)}...")
                    
                    # Process each image batch
                    batch_futures = {
                        executor.submit(self._process_single_image, image_url, input_face_encoding, session): image_url 
                        for image_url in image_batch
                    }
                    
                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(batch_futures):
                        image_url = batch_futures[future]
                        try:
                            match_found, match_score, face_location = future.result()
                            if match_found:
                                face_matches_found += 1
                                face_match_results.append({
                                    'url': image_url,
                                    'score': match_score,
                                    'location': face_location
                                })
                        except Exception:
                            # Skip failed images
                            pass
            
            print(f"✅ Image analysis complete: Found {face_matches_found} potential face matches")
        else:
            print("⚠️ Skipped image analysis - no input face encoding available")
        
        # Step 5: Compile and organize results
        analysis_end_time = time.time()
        total_time = analysis_end_time - start_time
        
        # Sort face matches by confidence (lower distance = better match)
        if face_match_results:
            face_match_results.sort(key=lambda x: x['score'])
        
        # Prepare final report
        final_report = {
            'person_name': person_name,
            'analysis_duration': f"{total_time:.2f} seconds",
            'face_analysis': face_data,
            'personal_information': personal_info,
            'face_matches': face_match_results[:10],  # Top 10 matches
            'search_results': {
                'total_count': len(all_results),
                'top_results': all_results[:15]  # Top 15 search results
            }
        }
        
        # Display summary
        self._display_report_summary(final_report)
        
        return final_report
    
    def _process_single_image(self, image_url, reference_encoding, session):
        """Process a single image for face matching"""
        try:
            face_location, face_encoding = self.face_analyzer.download_and_analyze_image(image_url, session)
            
            if face_location is None or face_encoding is None:
                return False, 1.0, None
            
            # Compare face with reference face
            match_found, match_score = self.face_analyzer.compare_face_encodings(
                reference_encoding, face_encoding, tolerance=0.6
            )
            
            return match_found, match_score, face_location
            
        except Exception:
            return False, 1.0, None
    
    def _cleanup_extracted_info(self, extracted_info, person_name):
        """Clean up and prioritize extracted information"""
        result = {}
        
        # Transfer and clean basic information
        if 'emails' in extracted_info and extracted_info['emails']:
            result['emails'] = extracted_info['emails'][:3]  # Top 3 emails
            
        if 'phones' in extracted_info and extracted_info['phones']:
            result['phones'] = extracted_info['phones'][:2]  # Top 2 phone numbers
            
        # Process social media accounts
        if 'social_media' in extracted_info and extracted_info['social_media']:
            result['social_media'] = {}
            for platform, usernames in extracted_info['social_media'].items():
                if usernames:
                    # Filter usernames that are likely to be related to the person
                    name_parts = person_name.lower().split()
                    filtered_usernames = [
                        u for u in usernames 
                        if any(part in u.lower() for part in name_parts) or len(u) > 3
                    ]
                    if filtered_usernames:
                        result['social_media'][platform] = filtered_usernames[:2]  # Top 2 usernames per platform
        
        # Process location data
        if 'locations' in extracted_info and extracted_info['locations']:
            result['locations'] = []
            seen_locations = set()
            for location in extracted_info['locations']:
                # Normalize location string for deduplication
                norm_location = ' '.join(location.lower().split())
                if norm_location not in seen_locations and len(norm_location) > 3:
                    result['locations'].append(location)
                    seen_locations.add(norm_location)
                    if len(result['locations']) >= 3:  # Top 3 locations
                        break
        
        # Process date of birth
        if 'birth_dates' in extracted_info and extracted_info['birth_dates']:
            result['birth_dates'] = extracted_info['birth_dates'][:1]  # Most likely DOB
            
        # Process employers
        if 'employers' in extracted_info and extracted_info['employers']:
            result['employers'] = []
            seen_employers = set()
            for employer in extracted_info['employers']:
                norm_employer = ' '.join(employer.lower().split())
                if norm_employer not in seen_employers and len(norm_employer) > 3:
                    result['employers'].append(employer)
                    seen_employers.add(norm_employer)
                    if len(result['employers']) >= 2:  # Top 2 employers
                        break
        
        # Process job titles
        if 'job_titles' in extracted_info and extracted_info['job_titles']:
            result['job_titles'] = []
            seen_titles = set()
            for title in extracted_info['job_titles']:
                norm_title = ' '.join(title.lower().split())
                if norm_title not in seen_titles and len(norm_title) > 3:
                    result['job_titles'].append(title)
                    seen_titles.add(norm_title)
                    if len(result['job_titles']) >= 2:  # Top 2 job titles
                        break
        
        # Process education
        if 'education' in extracted_info and extracted_info['education']:
            result['education'] = []
            seen_education = set()
            for edu in extracted_info['education']:
                norm_edu = ' '.join(edu.lower().split())
                if norm_edu not in seen_education and len(norm_edu) > 3:
                    result['education'].append(edu)
                    seen_education.add(norm_edu)
                    if len(result['education']) >= 2:  # Top 2 education entries
                        break
        
        # Process interests
        if 'interests' in extracted_info and extracted_info['interests']:
            result['interests'] = []
            seen_interests = set()
            for interest in extracted_info['interests']:
                norm_interest = ' '.join(interest.lower().split())
                if norm_interest not in seen_interests and len(norm_interest) > 3:
                    result['interests'].append(interest)
                    seen_interests.add(norm_interest)
                    if len(result['interests']) >= 3:  # Top 3 interests
                        break
        
        return result
    
    def _display_report_summary(self, report):
        """Display a summary of the OSINT analysis report"""
        print("\n" + "=" * 60)
        print(f"OSINT ANALYSIS SUMMARY FOR: {report['person_name']}")
        print("=" * 60)
        
        # Display personal information
        print("\nPERSONAL INFORMATION:")
        personal_info = report['personal_information']
        if not personal_info:
            print("  No personal information found")
        else:
            # Display emails
            if 'emails' in personal_info and personal_info['emails']:
                print(f"  Emails: {', '.join(personal_info['emails'])}")
            
            # Display phone numbers
            if 'phones' in personal_info and personal_info['phones']:
                print(f"  Phone Numbers: {', '.join(personal_info['phones'])}")
            
            # Display locations
            if 'locations' in personal_info and personal_info['locations']:
                print(f"  Locations: {', '.join(personal_info['locations'])}")
            
            # Display birth dates
            if 'birth_dates' in personal_info and personal_info['birth_dates']:
                print(f"  Date of Birth: {personal_info['birth_dates'][0]}")
            
            # Display employers
            if 'employers' in personal_info and personal_info['employers']:
                print(f"  Employers: {', '.join(personal_info['employers'])}")
            
            # Display job titles
            if 'job_titles' in personal_info and personal_info['job_titles']:
                print(f"  Job Titles: {', '.join(personal_info['job_titles'])}")
            
            # Display education
            if 'education' in personal_info and personal_info['education']:
                print(f"  Education: {', '.join(personal_info['education'])}")
            
            # Display social media
            if 'social_media' in personal_info and personal_info['social_media']:
                print("  Social Media:")
                for platform, usernames in personal_info['social_media'].items():
                    print(f"    - {platform.capitalize()}: {', '.join(usernames)}")
        
        # Display face matching results
        face_matches = report['face_matches']
        print("\nFACE MATCHING RESULTS:")
        if not face_matches:
            print("  No face matches found")
        else:
            print(f"  Found {len(face_matches)} potential matches")
            for i, match in enumerate(face_matches[:5], 1):  # Display top 5
                confidence = 100 * (1 - match['score'])  # Convert distance to confidence percentage
                print(f"  {i}. Match Confidence: {confidence:.1f}%")
        
        # Display search results summary
        print("\nSEARCH RESULTS SUMMARY:")
        print(f"  Total results found: {report['search_results']['total_count']}")
        
        # Display analysis duration
        print(f"\nAnalysis completed in {report['analysis_duration']}")
        print("=" * 60)
        
    def analyze_person(self, person_name, image_path):
        """Public method to run the full OSINT analysis and return results"""
        try:
            print(f"Starting OSINT analysis for {person_name}...")
            
            # Check if the image exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Calculate MD5 hash of the input for caching
            input_hash = hashlib.md5(f"{person_name}:{image_path}".encode()).hexdigest()
            cache_path = f"osint_cache_{input_hash}.json"
            
            # Check if we have cached results
            if os.path.exists(cache_path):
                print("Found cached results. Loading...")
                with open(cache_path, 'r') as f:
                    import json
                    cached_results = json.load(f)
                print("Using cached analysis results")
                self._display_report_summary(cached_results)
                return cached_results
            
            # Run the full analysis
            results = self.run(person_name, image_path)
            
            # Cache the results
            with open(cache_path, 'w') as f:
                import json
                json.dump(results, f)
            
            return results
            
        except Exception as e:
            print(f"Error in OSINT analysis: {str(e)}")
            return {
                'error': str(e),
                'person_name': person_name,
                'status': 'failed'
            }


# Example usage
if __name__ == "__main__":
    tool = OSINTTool()
    
    # Example: Search for a person by name and face image
    person_name = "vannesa octavia"  # Replace with actual name
    face_image = "face_example.jpg"  # Replace with actual image path
    
    # Run the OSINT analysis
    results = tool.analyze_person(person_name, face_image)
    
    # Results contain structured data that can be used programmatically
    print(f"Analysis complete for {person_name}")

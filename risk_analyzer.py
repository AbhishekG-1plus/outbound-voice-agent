#!/usr/bin/env python3
"""
Customer Risk Analysis Script for Debt Collection Transcripts (Hindi/Hinglish Support)

This script analyzes conversation transcripts from debt collection calls
and predicts customer risk levels based on various factors including:
- Sentiment analysis (English, Hindi, Hinglish)
- Keyword detection (multilingual)
- Response patterns
- Conversation flow
- Language switching patterns
- Hinglish-specific behavioral indicators

Supports:
- English keywords and phrases
- Hindi keywords (Devanagari script)
- Romanized Hindi (Hinglish)
- Mixed language conversations
- Cultural context-aware risk assessment
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

@dataclass
class RiskAnalysis:
    risk_level: RiskLevel
    risk_score: float  # 0-100
    sentiment_score: float  # -1 to 1
    cooperation_score: float  # 0-100
    key_indicators: List[str]
    recommendations: List[str]
    transcript_file: str
    analysis_timestamp: str

class CustomerRiskAnalyzer:
    """Analyze customer risk from debt collection call transcripts"""
    
    def __init__(self):
        # Risk indicators - keywords that suggest different risk levels
        # English keywords
        self.high_risk_keywords = [
            "can't pay", "no money", "broke", "unemployed", "lost job",
            "bankruptcy", "lawyer", "dispute", "wrong", "not mine",
            "never received", "scam", "harassment", "sue", "court",
            "refuse", "won't pay", "can't afford", "financial hardship",
            # Hindi keywords (Devanagari)
            "पैसे नहीं", "पैसा नहीं", "बेरोजगार", "नौकरी नहीं", "गलत", "मेरा नहीं",
            "नहीं मिला", "धोखा", "गलत नंबर", "वकील", "अदालत", "केस", "मना करता",
            # Romanized Hindi/Hinglish
            "paisa nahi", "paise nahi", "berozgar", "naukri nahi", "galat hai",
            "mera nahi", "nahi mila", "dhoka", "galat number", "vakeel", "adalat",
            "case karunga", "mana karta", "afford nahi kar sakta", "bankruptcy",
            "paisa khatam", "financial problem", "court jaaunga", "lawyer se baat"
        ]
        
        self.medium_risk_keywords = [
            "difficult", "tight", "struggling", "need time", "extension",
            "payment plan", "partial", "later", "next month", "busy",
            "forgot", "remind me", "will try", "maybe", "not sure",
            # Hindi keywords (Devanagari)
            "मुश्किल", "परेशानी", "समय चाहिए", "भूल गया", "बाद में", "अगले महीने",
            "व्यस्त", "कोशिश करूंगा", "शायद", "पता नहीं", "थोड़ा समय",
            # Romanized Hindi/Hinglish
            "mushkil", "pareshani", "samay chahiye", "bhul gaya", "baad mein",
            "agle mahine", "vyast", "busy hun", "koshish karunga", "shayad",
            "pata nahi", "thoda samay", "time chahiye", "extension chahiye",
            "payment plan", "partial payment", "installment mein", "emi mein"
        ]
        
        self.low_risk_keywords = [
            "yes", "okay", "sure", "will pay", "today", "tomorrow",
            "understand", "sorry", "apologize", "thank you", "appreciate",
            "payment", "pay now", "confirm", "agreed", "right away",
            # Hindi keywords (Devanagari)
            "हाँ", "ठीक", "समझ गया", "माफ़ करो", "धन्यवाद", "शुक्रिया", "सही",
            "आज", "कल", "अभी", "तुरंत", "पेमेंट", "भुगतान", "देता हूँ",
            # Romanized Hindi/Hinglish
            "haan", "theek", "theek hai", "samjh gaya", "maaf karo", "dhanyawad",
            "shukriya", "thank you", "sahi", "aaj", "kal", "abhi", "turant",
            "payment kar dunga", "paisa de dunga", "bhugtan", "pay kar dunga",
            "samay pe dunga", "confirm", "agreed", "bilkul", "zaroor"
        ]
        
        self.cooperation_indicators = [
            "thank you", "sorry", "understand", "appreciate", "yes",
            "okay", "sure", "will do", "agreed", "right",
            # Hindi/Hinglish cooperation indicators
            "haan", "theek", "theek hai", "samjh gaya", "maaf karo", "shukriya",
            "dhanyawad", "bilkul", "zaroor", "kar dunga", "de dunga", "samjha",
            "acha", "sahi", "ठीक", "हाँ", "समझ गया", "माफ़ करो", "धन्यवाद",
            "शुक्रिया", "बिल्कुल", "जरूर", "अच्छा", "सही"
        ]
        
        self.non_cooperation_indicators = [
            "no", "can't", "won't", "refuse", "busy", "later", "maybe",
            "not now", "call back", "don't have time", "not interested",
            # Hindi/Hinglish non-cooperation indicators
            "nahi", "nahin", "mana", "vyast", "busy", "baad mein", "phone rakh",
            "time nahi", "interested nahi", "pareshaan mat karo", "tang mat karo",
            "नहीं", "मना", "व्यस्त", "बाद में", "फोन रख", "समय नहीं", "परेशान मत करो",
            "तंग मत करो", "call back", "later call karo", "abhi nahi"
        ]

    def analyze_transcript(self, transcript_file: str) -> RiskAnalysis:
        """Analyze a single transcript file and return risk assessment"""
        
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
        except Exception as e:
            raise ValueError(f"Error reading transcript file {transcript_file}: {e}")
        
        # Extract conversation items
        items = transcript_data.get('items', [])
        
        # Separate user and assistant messages
        user_messages = []
        assistant_messages = []
        
        for item in items:
            if item.get('type') == 'message':
                content = ' '.join(item.get('content', [])).lower()
                if item.get('role') == 'user':
                    user_messages.append({
                        'content': content,
                        'interrupted': item.get('interrupted', False),
                        'confidence': item.get('transcript_confidence', 1.0)
                    })
                elif item.get('role') == 'assistant':
                    assistant_messages.append({
                        'content': content,
                        'interrupted': item.get('interrupted', False)
                    })
        
        # Perform analysis
        sentiment_score = self._analyze_sentiment(user_messages)
        cooperation_score = self._analyze_cooperation(user_messages)
        keyword_risk_score = self._analyze_keywords(user_messages)
        conversation_flow_score = self._analyze_conversation_flow(user_messages, assistant_messages)
        
        # Calculate overall risk score (0-100)
        risk_score = self._calculate_risk_score(
            sentiment_score, cooperation_score, keyword_risk_score, conversation_flow_score
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(risk_score)
        
        # Identify key indicators
        key_indicators = self._identify_key_indicators(user_messages)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(risk_level, key_indicators, cooperation_score)
        
        return RiskAnalysis(
            risk_level=risk_level,
            risk_score=risk_score,
            sentiment_score=sentiment_score,
            cooperation_score=cooperation_score,
            key_indicators=key_indicators,
            recommendations=recommendations,
            transcript_file=transcript_file,
            analysis_timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )

    def _analyze_sentiment(self, user_messages: List[Dict]) -> float:
        """Analyze sentiment of user messages (-1 to 1, where -1 is very negative)"""
        if not user_messages:
            return 0.0
        
        total_sentiment = 0.0
        message_count = 0
        
        # Enhanced sentiment analysis for Hindi/Hinglish
        positive_words = [
            # English positive words
            "thank", "sorry", "appreciate", "understand", "yes", "okay", "good", "fine",
            "great", "excellent", "wonderful", "happy", "pleased", "satisfied",
            # Hindi/Hinglish positive words (Romanized)
            "dhanyawad", "shukriya", "maaf karo", "samjh gaya", "theek", "acha", "badhiya",
            "khushi", "santushti", "prasanna", "haan", "bilkul", "zaroor", "accha",
            "sahi", "badiya", "mast", "sundar", "samjha", "theek hai",
            # Hindi positive words (Devanagari)
            "धन्यवाद", "शुक्रिया", "माफ़ करो", "समझ गया", "ठीक", "अच्छा", "बढ़िया",
            "खुशी", "संतुष्टि", "प्रसन्न", "हाँ", "बिल्कुल", "जरूर", "सही"
        ]
        
        negative_words = [
            # English negative words
            "no", "can't", "won't", "angry", "upset", "wrong", "bad", "hate", "refuse",
            "terrible", "awful", "horrible", "disgusted", "furious", "annoyed",
            # Hindi/Hinglish negative words (Romanized)
            "nahi", "nahin", "gussa", "pareshaan", "galat", "bura", "nafrat", "mana",
            "tang", "irritate", "problem", "mushkil", "takleef", "dukh", "ghussa",
            "pareshan", "khafa", "chid", "badtameez", "bakwas",
            # Hindi negative words (Devanagari)
            "नहीं", "गुस्सा", "परेशान", "गलत", "बुरा", "नफरत", "मना", "तंग",
            "समस्या", "मुश्किल", "तकलीफ", "दुःख", "खफा", "चिढ़", "बदतमीज", "बकवास"
        ]
        
        for msg in user_messages:
            content = msg['content']
            positive_count = sum(1 for word in positive_words if word in content)
            negative_count = sum(1 for word in negative_words if word in content)
            
            if positive_count > 0 or negative_count > 0:
                msg_sentiment = (positive_count - negative_count) / max(positive_count + negative_count, 1)
                total_sentiment += msg_sentiment
                message_count += 1
        
        return total_sentiment / max(message_count, 1)

    def _analyze_cooperation(self, user_messages: List[Dict]) -> float:
        """Analyze cooperation level (0-100)"""
        if not user_messages:
            return 50.0
        
        cooperation_points = 0
        total_indicators = 0
        
        for msg in user_messages:
            content = msg['content']
            
            # Check for cooperation indicators
            for indicator in self.cooperation_indicators:
                if indicator in content:
                    cooperation_points += 1
                    total_indicators += 1
            
            # Check for non-cooperation indicators
            for indicator in self.non_cooperation_indicators:
                if indicator in content:
                    cooperation_points -= 1
                    total_indicators += 1
        
        if total_indicators == 0:
            return 50.0  # Neutral if no clear indicators
        
        # Convert to 0-100 scale
        cooperation_ratio = cooperation_points / total_indicators
        return max(0, min(100, 50 + (cooperation_ratio * 50)))

    def _analyze_keywords(self, user_messages: List[Dict]) -> float:
        """Analyze risk based on keywords (0-100, higher = more risk)"""
        if not user_messages:
            return 0.0
        
        all_content = ' '.join(msg['content'] for msg in user_messages)
        
        high_risk_count = sum(1 for keyword in self.high_risk_keywords if keyword in all_content)
        medium_risk_count = sum(1 for keyword in self.medium_risk_keywords if keyword in all_content)
        low_risk_count = sum(1 for keyword in self.low_risk_keywords if keyword in all_content)
        
        # Calculate weighted risk score
        risk_score = (high_risk_count * 30) + (medium_risk_count * 15) - (low_risk_count * 10)
        
        # Normalize to 0-100 scale
        return max(0, min(100, risk_score))

    def _analyze_conversation_flow(self, user_messages: List[Dict], assistant_messages: List[Dict]) -> float:
        """Analyze conversation flow patterns (0-100, higher = more risk)"""
        risk_score = 0.0
        
        # Check if user interrupted agent frequently
        user_interruptions = sum(1 for msg in user_messages if msg.get('interrupted', False))
        if user_interruptions > 2:
            risk_score += 20
        
        # Check if user responses are very short (might indicate disengagement)
        short_responses = sum(1 for msg in user_messages if len(msg['content'].split()) <= 2)
        if len(user_messages) > 0 and short_responses / len(user_messages) > 0.7:
            risk_score += 15
        
        # Check transcript confidence (low confidence might indicate unclear speech/agitation)
        low_confidence_count = sum(1 for msg in user_messages 
                                 if msg.get('confidence', 1.0) < 0.7)
        if len(user_messages) > 0 and low_confidence_count / len(user_messages) > 0.5:
            risk_score += 10
        
        return min(100, risk_score)

    def _calculate_risk_score(self, sentiment: float, cooperation: float, 
                            keyword_risk: float, flow_risk: float) -> float:
        """Calculate overall risk score (0-100)"""
        
        # Convert sentiment to risk (negative sentiment = higher risk)
        sentiment_risk = max(0, (1 - sentiment) * 50)  # -1 sentiment = 100 risk, +1 sentiment = 0 risk
        
        # Convert cooperation to risk (low cooperation = higher risk)
        cooperation_risk = 100 - cooperation
        
        # Weighted average of all factors
        weights = {
            'sentiment': 0.25,
            'cooperation': 0.35,
            'keywords': 0.30,
            'flow': 0.10
        }
        
        risk_score = (
            sentiment_risk * weights['sentiment'] +
            cooperation_risk * weights['cooperation'] +
            keyword_risk * weights['keywords'] +
            flow_risk * weights['flow']
        )
        
        return min(100, max(0, risk_score))

    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on score"""
        if risk_score >= 75:
            return RiskLevel.CRITICAL
        elif risk_score >= 50:
            return RiskLevel.HIGH
        elif risk_score >= 25:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _identify_key_indicators(self, user_messages: List[Dict]) -> List[str]:
        """Identify key risk indicators found in the conversation"""
        indicators = []
        all_content = ' '.join(msg['content'] for msg in user_messages)
        
        # Check for specific indicators
        for keyword in self.high_risk_keywords:
            if keyword in all_content:
                # Translate common Hindi phrases for better reporting
                translated_keyword = self._translate_keyword_for_report(keyword)
                indicators.append(f"High risk keyword: '{translated_keyword}'")
        
        for keyword in self.medium_risk_keywords:
            if keyword in all_content:
                translated_keyword = self._translate_keyword_for_report(keyword)
                indicators.append(f"Medium risk keyword: '{translated_keyword}'")
        
        # Check for Hinglish-specific patterns
        hinglish_patterns = self._detect_hinglish_patterns(all_content)
        indicators.extend(hinglish_patterns)
        
        # Check for patterns
        if len(user_messages) == 0:
            indicators.append("No user responses - possible call avoidance")
        elif len(user_messages) == 1 and len(user_messages[0]['content'].split()) <= 2:
            indicators.append("Very brief response - possible disengagement")
        
        # Check for language switching patterns (risk indicator in debt collection)
        if self._detect_language_switching(user_messages):
            indicators.append("Frequent language switching - possible avoidance tactic")
        
        return indicators[:7]  # Increased limit for more detailed analysis
    
    def _translate_keyword_for_report(self, keyword: str) -> str:
        """Translate Hindi keywords to English for better report readability"""
        translations = {
            # Hindi to English translations for reporting
            "पैसे नहीं": "no money",
            "पैसा नहीं": "no money", 
            "बेरोजगार": "unemployed",
            "नौकरी नहीं": "no job",
            "गलत": "wrong",
            "मेरा नहीं": "not mine",
            "नहीं मिला": "never received",
            "धोखा": "fraud/scam",
            "वकील": "lawyer",
            "अदालत": "court",
            "केस": "case",
            "मुश्किल": "difficult",
            "परेशानी": "problem/trouble",
            "समय चाहिए": "need time",
            "भूल गया": "forgot",
            "बाद में": "later",
            "व्यस्त": "busy",
            "हाँ": "yes",
            "ठीक": "okay",
            "समझ गया": "understood",
            "माफ़ करो": "sorry",
            "धन्यवाद": "thank you",
            "शुक्रिया": "thank you",
            "नहीं": "no",
            "गुस्सा": "angry",
            "परेशान": "upset/troubled"
        }
        return translations.get(keyword, keyword)
    
    def _detect_hinglish_patterns(self, content: str) -> List[str]:
        """Detect specific Hinglish patterns that might indicate risk"""
        patterns = []
        
        # Common evasive Hinglish phrases
        evasive_phrases = [
            "abhi busy hun", "time nahi hai", "baad mein call karo", 
            "pareshaan mat karo", "tang mat karo", "galat number hai",
            "mujhe pata nahi", "kuch nahi pata", "samjh nahi aaya"
        ]
        
        for phrase in evasive_phrases:
            if phrase in content:
                patterns.append(f"Evasive Hinglish phrase: '{phrase}'")
        
        # Aggressive/hostile Hinglish patterns
        hostile_phrases = [
            "phone rakh", "bakwas mat karo", "jhooth bol rahe ho",
            "scam hai ye", "fraud company", "police complaint karunga"
        ]
        
        for phrase in hostile_phrases:
            if phrase in content:
                patterns.append(f"Hostile Hinglish phrase: '{phrase}'")
        
        # Financial distress Hinglish indicators
        distress_phrases = [
            "paisa nahi hai", "afford nahi kar sakta", "salary nahi aayi",
            "job chali gayi", "business band ho gaya", "EMI bhi nahi de pa raha"
        ]
        
        for phrase in distress_phrases:
            if phrase in content:
                patterns.append(f"Financial distress indicator: '{phrase}'")
        
        return patterns
    
    def _detect_language_switching(self, user_messages: List[Dict]) -> bool:
        """Detect if user frequently switches between Hindi and English"""
        if len(user_messages) < 3:
            return False
        
        # Simple heuristic: check if messages alternate between having Hindi and English words
        hindi_indicators = ["nahi", "hai", "kar", "se", "mein", "ko", "ka", "ki", "ke"]
        english_indicators = ["the", "and", "is", "are", "can", "will", "have", "not"]
        
        language_pattern = []
        for msg in user_messages:
            content = msg['content']
            hindi_count = sum(1 for word in hindi_indicators if word in content)
            english_count = sum(1 for word in english_indicators if word in content)
            
            if hindi_count > english_count:
                language_pattern.append('H')  # Hindi dominant
            elif english_count > hindi_count:
                language_pattern.append('E')  # English dominant
            else:
                language_pattern.append('M')  # Mixed or neutral
        
        # Check for frequent switching (more than 50% of transitions are switches)
        switches = 0
        for i in range(len(language_pattern) - 1):
            if language_pattern[i] != language_pattern[i + 1] and language_pattern[i] != 'M' and language_pattern[i + 1] != 'M':
                switches += 1
        
        return switches > len(language_pattern) * 0.3  # More than 30% switches

    def _generate_recommendations(self, risk_level: RiskLevel, indicators: List[str], 
                                cooperation_score: float) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        if risk_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "Escalate to senior collection specialist immediately",
                "Consider legal action or external collection agency",
                "Document all interactions thoroughly",
                "Review account for potential disputes or fraud"
            ])
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "Schedule follow-up call within 24-48 hours",
                "Offer payment plan options",
                "Consider supervisor review of account",
                "Document customer concerns and objections"
            ])
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.extend([
                "Follow up within 1 week",
                "Send payment reminder via email/SMS",
                "Monitor account closely for changes",
                "Be prepared to offer flexible payment options"
            ])
        else:  # LOW risk
            recommendations.extend([
                "Standard follow-up schedule",
                "Customer appears cooperative",
                "Continue with normal collection process",
                "Consider this a positive interaction"
            ])
        
        # Add cooperation-specific recommendations
        if cooperation_score < 30:
            recommendations.append("Customer showed low cooperation - consider different approach")
        elif cooperation_score > 70:
            recommendations.append("Customer was cooperative - maintain positive relationship")
        
        # Add Hinglish-specific recommendations
        hinglish_recommendations = self._get_hinglish_recommendations(indicators)
        recommendations.extend(hinglish_recommendations)
        
        return recommendations
    
    def _get_hinglish_recommendations(self, indicators: List[str]) -> List[str]:
        """Generate Hinglish-specific recommendations based on detected patterns"""
        recommendations = []
        
        # Check for specific Hinglish patterns in indicators
        indicator_text = ' '.join(indicators).lower()
        
        if "evasive hinglish phrase" in indicator_text:
            recommendations.append("Customer using evasive Hinglish - try direct Hindi approach")
        
        if "hostile hinglish phrase" in indicator_text:
            recommendations.append("Customer showing hostility in Hinglish - escalate to Hindi-speaking senior agent")
        
        if "financial distress indicator" in indicator_text:
            recommendations.append("Financial distress detected - offer EMI/installment options in Hindi")
        
        if "language switching" in indicator_text:
            recommendations.append("Customer switching languages - may indicate discomfort, use consistent Hindi")
        
        if any("no money" in ind or "unemployed" in ind or "no job" in ind for ind in indicators):
            recommendations.append("Financial hardship indicated - consider compassionate collection approach")
        
        if any("lawyer" in ind or "court" in ind or "case" in ind for ind in indicators):
            recommendations.append("Legal threats detected - document thoroughly and involve legal team")
        
        return recommendations

    def analyze_all_transcripts(self, transcript_dir: str = "transcripts") -> List[RiskAnalysis]:
        """Analyze all transcript files in the directory"""
        if not os.path.exists(transcript_dir):
            raise ValueError(f"Transcript directory '{transcript_dir}' does not exist")
        
        analyses = []
        transcript_files = [f for f in os.listdir(transcript_dir) if f.endswith('.json')]
        
        for filename in transcript_files:
            filepath = os.path.join(transcript_dir, filename)
            try:
                analysis = self.analyze_transcript(filepath)
                analyses.append(analysis)
            except Exception as e:
                print(f"Error analyzing {filename}: {e}")
        
        return analyses

    def generate_report(self, analyses: List[RiskAnalysis], output_file: str = None) -> str:
        """Generate a comprehensive risk analysis report"""
        if not analyses:
            return "No analyses to report"
        
        report = []
        report.append("=" * 60)
        report.append("CUSTOMER RISK ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Transcripts Analyzed: {len(analyses)}")
        report.append("")
        
        # Summary statistics
        risk_counts = {level: 0 for level in RiskLevel}
        total_risk_score = 0
        
        for analysis in analyses:
            risk_counts[analysis.risk_level] += 1
            total_risk_score += analysis.risk_score
        
        avg_risk_score = total_risk_score / len(analyses)
        
        report.append("SUMMARY STATISTICS:")
        report.append("-" * 20)
        for level in RiskLevel:
            count = risk_counts[level]
            percentage = (count / len(analyses)) * 100
            report.append(f"{level.value} Risk: {count} ({percentage:.1f}%)")
        report.append(f"Average Risk Score: {avg_risk_score:.1f}/100")
        report.append("")
        
        # Individual analyses
        report.append("INDIVIDUAL ANALYSES:")
        report.append("-" * 20)
        
        for analysis in sorted(analyses, key=lambda x: x.risk_score, reverse=True):
            report.append(f"\nFile: {os.path.basename(analysis.transcript_file)}")
            report.append(f"Risk Level: {analysis.risk_level.value}")
            report.append(f"Risk Score: {analysis.risk_score:.1f}/100")
            report.append(f"Sentiment: {analysis.sentiment_score:.2f}")
            report.append(f"Cooperation: {analysis.cooperation_score:.1f}%")
            
            if analysis.key_indicators:
                report.append("Key Indicators:")
                for indicator in analysis.key_indicators:
                    report.append(f"  • {indicator}")
            
            if analysis.recommendations:
                report.append("Recommendations:")
                for rec in analysis.recommendations:
                    report.append(f"  • {rec}")
            
            report.append("-" * 40)
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Report saved to: {output_file}")
        
        return report_text

def main():
    """Main function to run the risk analysis"""
    analyzer = CustomerRiskAnalyzer()
    
    try:
        # Analyze all transcripts
        print("🔍 Analyzing transcripts...")
        analyses = analyzer.analyze_all_transcripts()
        
        if not analyses:
            print("❌ No transcript files found in 'transcripts' directory")
            return
        
        print(f"✅ Analyzed {len(analyses)} transcripts")
        
        # Generate report
        report_filename = f"risk_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report = analyzer.generate_report(analyses, report_filename)
        
        # Print summary to console
        print("\n" + "=" * 50)
        print("RISK ANALYSIS SUMMARY")
        print("=" * 50)
        
        risk_counts = {level: 0 for level in RiskLevel}
        for analysis in analyses:
            risk_counts[analysis.risk_level] += 1
        
        for level in RiskLevel:
            count = risk_counts[level]
            if count > 0:
                print(f"{level.value} Risk: {count} customers")
        
        print(f"\n📄 Full report saved to: {report_filename}")
        
        # Show high-risk customers
        high_risk = [a for a in analyses if a.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        if high_risk:
            print(f"\n⚠️  HIGH PRIORITY: {len(high_risk)} customers need immediate attention:")
            for analysis in high_risk:
                filename = os.path.basename(analysis.transcript_file)
                print(f"  • {filename} - {analysis.risk_level.value} ({analysis.risk_score:.1f}/100)")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

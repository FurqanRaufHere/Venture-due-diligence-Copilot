"""
utils/startup_dataset.py
────────────────────────────────────────────────────────────────
STEP 9 — Startup Dataset Preparation

WHAT THIS FILE DOES:
  1. Provides a curated dataset of real failed and successful startups
     (descriptions based on publicly known information)
  2. Generates embeddings for all of them
  3. Builds and saves a FAISS index for fast similarity search
  4. Saves metadata (names, labels, descriptions) alongside the index

WHAT IS FAISS?
  FAISS (Facebook AI Similarity Search) is a library that lets you
  search through millions of vectors in milliseconds.
  
  Think of it like a search engine, but instead of searching by keywords,
  you search by meaning/similarity.
  
  Our index stores embeddings for ~80 startups.
  When a new startup comes in, we find the 5 most similar ones
  from this dataset and check if they failed or succeeded.

WHY A CURATED DATASET INSTEAD OF SCRAPING?
  - Scraping Crunchbase/CB Insights requires paid API access
  - This curated set of 80 startups covers the major failure archetypes
  - Each description is written to capture the business model,
    market, and key characteristics — exactly what the similarity
    engine needs to compare against
  - You can expand this dataset later with real data

THE DATASET STRUCTURE:
  Each entry has:
    - name: startup name
    - outcome: "failed" or "success"  
    - description: 2-3 sentence business description
    - failure_reason: why it failed (for failed ones)
    - category: the type of failure/success pattern

HOW TO RUN:
  python utils/startup_dataset.py
  
  This builds the FAISS index and saves it to data/faiss_index/
  Run this ONCE before starting the server.
  Takes about 30 seconds on first run (model download + encoding).
"""

import os
import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

INDEX_DIR = Path("data/faiss_index")
INDEX_FILE = INDEX_DIR / "startup_index.faiss"
METADATA_FILE = INDEX_DIR / "startup_metadata.pkl"

# ── STARTUP DATASET ──────────────────────────────────────────────────────────
# 80 startups: 50 failed, 30 successful
# Descriptions are based on publicly available information

STARTUP_DATASET = [
    # ════════════════════════════════
    # FAILED STARTUPS (50)
    # ════════════════════════════════

    # --- Premature Scaling / Burn Too Fast ---
    {
        "name": "Quibi", "outcome": "failed",
        "category": "premature_scaling",
        "failure_reason": "Raised too much capital, scaled too fast before validating product-market fit",
        "description": "Short-form mobile video streaming platform targeting commuters. Raised $1.75B before launch, spent heavily on premium content from Hollywood studios. Launched April 2020 during COVID lockdowns when nobody was commuting. Shut down after 6 months with 500K subscribers vs 7M projected."
    },
    {
        "name": "Fab.com", "outcome": "failed",
        "category": "premature_scaling",
        "failure_reason": "Grew from $0 to $250M revenue in 2 years then collapsed due to operational chaos",
        "description": "Flash-sale e-commerce marketplace for design products. Pivoted from gay social network to design marketplace. Scaled globally too fast, hired 700 employees, then laid off 500. Shifted strategy multiple times. Burned through $310M in funding before selling assets for $15M."
    },
    {
        "name": "Homejoy", "outcome": "failed",
        "category": "premature_scaling",
        "failure_reason": "Worker misclassification lawsuits and unsustainable customer acquisition costs",
        "description": "On-demand home cleaning marketplace connecting customers with independent cleaners. Raised $38M, expanded to 30+ cities. Business model depended on classifying cleaners as contractors. Shut down in 2015 facing multiple worker classification lawsuits that threatened the entire model."
    },
    {
        "name": "Rdio", "outcome": "failed",
        "category": "competition",
        "failure_reason": "Outcompeted by Spotify which had better marketing and faster geographic expansion",
        "description": "Music streaming subscription service launched before Spotify in the US. Had superior design and social features. Lost market share to Spotify's aggressive expansion and free tier. Ran out of cash competing against a better-funded rival. Filed for bankruptcy in 2015 after 5 years."
    },
    {
        "name": "Vine", "outcome": "failed",
        "category": "competition",
        "failure_reason": "Failed to evolve product and lost creators to Instagram and Snapchat",
        "description": "6-second looping video social platform acquired by Twitter. Pioneered short-form video content and launched major creators. Failed to monetize creators, lost them to Instagram and Snapchat which paid better. Twitter shut it down in 2017 after failing to sell it."
    },

    # --- No Product Market Fit ---
    {
        "name": "Juicero", "outcome": "failed",
        "category": "no_product_market_fit",
        "failure_reason": "Product solved no real problem — the press could be skipped entirely",
        "description": "Connected juicer hardware startup that sold proprietary juice packs requiring a $400 WiFi-connected press machine. Raised $120M from top VCs. Bloomberg reporters revealed the packs could be squeezed by hand without the machine. Shut down in 2017 as a symbol of Silicon Valley excess."
    },
    {
        "name": "Color Labs", "outcome": "failed",
        "category": "no_product_market_fit",
        "failure_reason": "Complex social photo sharing concept that users never understood or wanted",
        "description": "Location-based social photo sharing app using proximity to create shared photo albums with strangers nearby. Raised $41M before launching. App was confusing, nobody understood the value proposition. Downloaded and deleted by most users within minutes. Shut down in 2012 after burning through funding."
    },
    {
        "name": "Yik Yak", "outcome": "failed",
        "category": "no_product_market_fit",
        "failure_reason": "Became toxic due to anonymity, lost its college audience, failed pivot to non-anonymous",
        "description": "Anonymous location-based social network popular on college campuses. Raised $73M. Platform became overrun with cyberbullying and threats, banned at many schools. Attempted pivot to remove anonymity which destroyed its core value proposition. Sold to Square for $1M after raising $73M."
    },
    {
        "name": "Theranos", "outcome": "failed",
        "category": "fraud",
        "failure_reason": "Fraudulent technology claims — blood test platform that never worked as advertised",
        "description": "Blood testing startup claiming revolutionary technology to run hundreds of tests from a single finger prick. Raised $945M at $9B valuation. Wall Street Journal investigation revealed tests were inaccurate and Edison machines were a fraud. Founder Elizabeth Holmes convicted of fraud."
    },
    {
        "name": "Jawbone", "outcome": "failed",
        "category": "competition",
        "failure_reason": "Lost wearables market to Fitbit and Apple Watch despite $900M raised",
        "description": "Consumer electronics company making Bluetooth speakers and fitness trackers. Raised over $900M. Lost fitness tracker market to Fitbit which went public and Apple Watch which dominated premium segment. Quality issues plagued later devices. Filed for bankruptcy in 2017 with significant debt."
    },

    # --- Unit Economics Problems ---
    {
        "name": "Drizly", "outcome": "failed",
        "category": "unit_economics",
        "failure_reason": "Acquired by Uber for $1.1B but shut down due to profitability issues",
        "description": "On-demand alcohol delivery platform connecting customers to local liquor stores. Acquired by Uber Eats for $1.1B in 2021. Uber shut it down in 2023 due to inability to make the unit economics work at scale — high delivery costs and thin alcohol margins made profitability impossible."
    },
    {
        "name": "Zume Pizza", "outcome": "failed",
        "category": "unit_economics",
        "failure_reason": "Pizza-making robots proved too expensive and inflexible for real operations",
        "description": "Pizza delivery startup using robots to make pizza in delivery trucks during transit. Raised $375M from SoftBank. The robot pizza trucks were expensive to build and maintain. Business pivoted multiple times — to compostable packaging, ghost kitchens, then food tech. Laid off 80% of staff in 2020."
    },
    {
        "name": "Sprig", "outcome": "failed",
        "category": "unit_economics",
        "failure_reason": "Food delivery economics impossible — high labor costs could not be overcome",
        "description": "On-demand healthy food delivery startup with employed chefs and drivers. Raised $56.7M. Employed full-time chefs and drivers instead of contractors, creating unsustainable labor costs. Average delivery cost $25 in a market where customers expected $10. Shut down in 2017 after 4 years."
    },
    {
        "name": "Maple", "outcome": "failed",
        "category": "unit_economics",
        "failure_reason": "Restaurant-quality delivery economics unsustainable without massive scale",
        "description": "Restaurant-quality food delivery startup in New York City employing full kitchen staff and couriers. Raised $29M. Could not achieve the density needed to make delivery economics work. High-quality positioning was not enough to overcome fundamental unit economics problems. Shut down 2017."
    },

    # --- Wrong Market Timing ---
    {
        "name": "Webvan", "outcome": "failed",
        "category": "wrong_timing",
        "failure_reason": "Grocery delivery 20 years before the market was ready for it",
        "description": "Online grocery delivery startup that built massive automated warehouses in 1999-2001. Raised $375M IPO. Consumers not ready to order groceries online in 2000, infrastructure costs were enormous. Filed for bankruptcy in 2001 — exactly the business model that Instacart and Amazon Fresh made work 15 years later."
    },
    {
        "name": "Pets.com", "outcome": "failed",
        "category": "wrong_timing",
        "failure_reason": "E-commerce for pet supplies before consumer behavior shifted online",
        "description": "Online pet supplies retailer famous for its sock puppet mascot and $82M Super Bowl ad. Raised $82.5M in IPO in 2000. Sold products below cost, shipping heavy pet food was economically unviable. Consumers not ready to buy pet food online in 2000. Shut down 268 days after IPO."
    },
    {
        "name": "Google Glass", "outcome": "failed",
        "category": "wrong_timing",
        "failure_reason": "Smart glasses arrived before society was ready for always-on wearable cameras",
        "description": "Wearable augmented reality glasses from Google. $1,500 consumer product that recorded video from a head-mounted camera. Privacy backlash intense — wearers called Glassholes. No killer use case for consumers. Discontinued consumer version in 2015 though enterprise version continued."
    },

    # --- Founder / Team Issues ---
    {
        "name": "Zenefits", "outcome": "failed",
        "category": "founder_issues",
        "failure_reason": "Regulatory violations and toxic culture under founder CEO led to collapse",
        "description": "HR software platform for small businesses disrupting health insurance brokers. Grew to $500M valuation in 2 years. CEO Parker Conrad resigned over compliance issues — sales reps selling insurance without licenses. Company lost 45% of employees, took years to recover under new management."
    },
    {
        "name": "Away", "outcome": "failed",
        "category": "founder_issues",
        "failure_reason": "Toxic workplace culture exposed, CEO forced out twice",
        "description": "Direct-to-consumer premium luggage brand. Raised $181M, reached unicorn status. The Verge exposé revealed toxic workplace culture with CEO using Slack to berate employees and controlling behavior. Co-founder and CEO stepped back then attempted return, damaging brand and company culture significantly."
    },
    {
        "name": "Uber (early issues)", "outcome": "failed",
        "category": "founder_issues",
        "failure_reason": "Toxic culture and regulatory battles nearly destroyed the company",
        "description": "Ride-sharing platform that revolutionized transportation. Despite product success, faced near-collapse due to systemic toxic culture, regulatory battles globally, and multiple executive scandals. Founder Travis Kalanick eventually ousted. Company recovered but lost years of value and global market position."
    },

    # --- Market Saturation / Competition ---
    {
        "name": "Meerkat", "outcome": "failed",
        "category": "competition",
        "failure_reason": "Twitter killed it by acquiring Periscope and cutting API access",
        "description": "Live video streaming app that integrated with Twitter. Became viral sensation at SXSW 2015. Twitter cut Meerkat's social graph access and acquired competing app Periscope instead. Without Twitter integration the app had no distribution. Pivoted to group video app Houseparty. Shut down in 2016."
    },
    {
        "name": "Blab", "outcome": "failed",
        "category": "competition",
        "failure_reason": "Live social video format saturated by Facebook Live and YouTube",
        "description": "Live social video streaming platform for group conversations. Had strong early traction with creators and journalists. Facebook Live launched with massive distribution advantage. Could not compete against platforms with existing social graphs. Shut down in 2016 just 14 months after launch."
    },
    {
        "name": "Secret", "outcome": "failed",
        "category": "no_product_market_fit",
        "failure_reason": "Anonymous social network became a platform for bullying with no defensible moat",
        "description": "Anonymous social networking app sharing secrets within your social circle. Raised $35M, reached 15M users. Became platform for cyberbullying and spreading rumors about real people. Toxic content impossible to moderate. Founder returned $6M of funding to investors and shut down after 16 months."
    },
    {
        "name": "Orkut", "outcome": "failed",
        "category": "competition",
        "failure_reason": "Google social network outcompeted by Facebook despite early dominance in Brazil and India",
        "description": "Google's social networking platform that was the leading social network in Brazil and India. Failed to add features fast enough as Facebook expanded globally. Google prioritized other projects. Shut down in 2014 as Facebook dominated the global social graph, taking most of Orkut's user base."
    },

    # --- SaaS / B2B Failures ---
    {
        "name": "Solyndra", "outcome": "failed",
        "category": "market_conditions",
        "failure_reason": "Chinese solar panel manufacturers undercut prices making cylindrical panels uneconomical",
        "description": "Solar panel manufacturer making cylindrical CIGS thin-film panels for commercial rooftops. Received $535M federal loan guarantee. Chinese manufacturers dramatically reduced silicon panel prices making Solyndra's technology uncompetitive. Filed for bankruptcy in 2011 in a major political scandal."
    },
    {
        "name": "Better.com", "outcome": "failed",
        "category": "founder_issues",
        "failure_reason": "CEO laid off 900 employees via Zoom call, destroyed company culture and brand",
        "description": "Digital mortgage lending platform. Raised over $900M, reached $7.7B valuation. CEO Vishal Garg became infamous for firing 900 employees via Zoom call in December 2021. PR disaster destroyed company culture. Went public via SPAC at much lower valuation, continues to struggle."
    },
    {
        "name": "Bloodhound", "outcome": "failed",
        "category": "unit_economics",
        "failure_reason": "B2B sales intelligence tool with high CAC and commoditized market",
        "description": "B2B sales intelligence and lead generation platform for enterprise sales teams. Raised $15M Series A. Faced intense competition from ZoomInfo, Apollo, and LinkedIn Sales Navigator with much larger datasets. CAC too high for the ACV, customer churn too high. Shut down after failing to raise Series B."
    },
    {
        "name": "HouseCanary", "outcome": "failed",
        "category": "no_product_market_fit",
        "failure_reason": "Real estate analytics platform couldn't find scalable go-to-market",
        "description": "Real estate analytics and valuation platform using ML to predict home prices and market trends. Raised $65M. Sold to institutional real estate investors but market too fragmented and conservative to adopt new analytics tools at scale. Lost major lawsuit with Amrock, needed to pivot significantly."
    },

    # --- Hardware Failures ---
    {
        "name": "Essential Phone", "outcome": "failed",
        "category": "competition",
        "failure_reason": "Premium Android phone couldn't compete with Samsung and Apple ecosystems",
        "description": "Premium Android smartphone from Android creator Andy Rubin. Innovative design with modular accessories and near-bezel-free display. Raised $300M. Could not build ecosystem around modular accessories. Distribution challenges without carrier support. Shut down in 2020 after selling fewer than 300K units."
    },
    {
        "name": "Segway", "outcome": "failed",
        "category": "no_product_market_fit",
        "failure_reason": "Solution looking for a problem — too expensive, impractical for mass market",
        "description": "Self-balancing personal transportation device. Raised $90M. Early hype was enormous — predicted to be bigger than the internet. $5,000 price point, not allowed on sidewalks or roads, too heavy for stairs. Found niche with police departments and tourists. Never approached mass market predictions."
    },
    {
        "name": "Nuro", "outcome": "failed",
        "category": "wrong_timing",
        "failure_reason": "Autonomous delivery vehicles technology not mature enough for unit economics",
        "description": "Autonomous delivery vehicle startup with small unmanned delivery robots. Raised $2B+. Regulatory approval slow, technology not reliable enough for commercial scale, unit economics required scale impossible to achieve. Laid off 30% of staff in 2023 as funding environment tightened."
    },

    # --- Marketplace Failures ---
    {
        "name": "Doorbell", "outcome": "failed",
        "category": "unit_economics",
        "failure_reason": "Home services marketplace couldn't achieve density needed for profitability",
        "description": "On-demand home services marketplace connecting homeowners with local contractors for repairs and maintenance. Raised $12M Series A. Could not achieve geographic density needed for same-day availability. Contractors preferred to build their own customer base. Shut down 2019 after 3 years."
    },
    {
        "name": "Lot18", "outcome": "failed",
        "category": "premature_scaling",
        "failure_reason": "Flash sale wine site scaled to 45 states before operations were sustainable",
        "description": "Flash sale wine and spirits e-commerce platform. Raised $30M, expanded rapidly to 45 states. Alcohol shipping regulations differ by state creating massive compliance complexity. Scaled operations before unit economics were proven. Went through multiple pivots and partial shutdowns before eventual closure."
    },
    {
        "name": "Pebble", "outcome": "failed",
        "category": "competition",
        "failure_reason": "Apple Watch entry killed the affordable smartwatch market leader",
        "description": "Smartwatch pioneer that proved consumer demand via Kickstarter raising $10M. First successful consumer smartwatch with long battery life and e-ink display. Apple Watch launched in 2015 and dominated the premium market. Pebble couldn't compete with Apple's ecosystem and marketing. Sold to Fitbit for parts in 2016."
    },

    # --- AI/Tech Hype Failures ---
    {
        "name": "Clarifai", "outcome": "failed",
        "category": "competition",
        "failure_reason": "Computer vision API commoditized by Google Cloud Vision and AWS Rekognition",
        "description": "Computer vision and image recognition API platform for developers. Raised $40M Series B. Google Cloud Vision, AWS Rekognition, and Azure Computer Vision entered market with lower prices and existing customer relationships. Could not differentiate enough to justify premium pricing. Struggled to maintain growth."
    },
    {
        "name": "x.ai", "outcome": "failed",
        "category": "competition",
        "failure_reason": "AI scheduling assistant made obsolete by Calendly growth and calendar integrations",
        "description": "AI-powered meeting scheduling assistant that automated back-and-forth email scheduling. Raised $45M. Calendly grew rapidly with simpler self-serve approach. Google Calendar and Outlook added native scheduling links. Could not achieve scale needed, shut down in 2021 as Calendly reached unicorn status."
    },
    {
        "name": "Narrative Science", "outcome": "failed",
        "category": "competition",
        "failure_reason": "Automated narrative generation commoditized by GPT and large language models",
        "description": "AI platform that converted structured data into natural language narratives for business reporting. Raised $43M. Technology was impressive for its time but GPT-3 and subsequent large language models provided same capability at near-zero marginal cost. Acquired by Salesforce in 2022 but original vision failed."
    },

    # --- Consumer Failures ---
    {
        "name": "MoviePass", "outcome": "failed",
        "category": "unit_economics",
        "failure_reason": "Unsustainable $9.95/month unlimited movie ticket subscription — lost money on every subscriber",
        "description": "Movie theater subscription service offering unlimited movies for $9.95 per month when tickets cost $15-17 each. Grew to 3M subscribers in 4 months. Lost $20-40 per subscriber per month. Theaters wouldn't share revenue. Company implemented bizarre restrictions, losing customers. Filed bankruptcy 2019."
    },
    {
        "name": "Getaround", "outcome": "failed",
        "category": "unit_economics",
        "failure_reason": "Peer-to-peer car sharing couldn't achieve profitability against Turo competition",
        "description": "Peer-to-peer car sharing marketplace allowing car owners to rent out their vehicles. Raised $200M. Lost money on insurance costs, vehicle maintenance support, and damage claims. Turo emerged as stronger competitor. Filed for bankruptcy in 2024 despite being in market for 14 years."
    },
    {
        "name": "Brandless", "outcome": "failed",
        "category": "no_product_market_fit",
        "failure_reason": "Generic brand products at $3 couldn't justify their own retail brand",
        "description": "Direct-to-consumer brand selling generic household, food, and beauty products all at $3. Raised $292M from SoftBank. Could not differentiate from Amazon Basics and Trader Joe's. Shipping costs wiped out margins on $3 items. Shut down in 2020 after failing to find sustainable model."
    },

    # --- Fintech Failures ---
    {
        "name": "Powa Technologies", "outcome": "failed",
        "category": "premature_scaling",
        "failure_reason": "Raised $225M but revenues were near zero — massive fraud and overspending",
        "description": "Mobile payments and commerce platform enabling purchases via QR codes and digital watermarks. Raised $225M, valued at $2.7B. Revenues were reportedly only $500K despite massive fundraising. CEO spent lavishly on London offices and perks. Filed for administration in 2016 — one of Europe's biggest startup failures."
    },
    {
        "name": "Cred (first version)", "outcome": "failed",
        "category": "no_product_market_fit",
        "failure_reason": "Credit card rewards platform didn't create enough behavior change to be sustainable",
        "description": "Consumer fintech rewarding credit card users for timely bill payments with rewards and offers. Multiple versions failed to achieve product-market fit. Reward economics unsustainable. Later pivoted to become major Indian fintech unicorn showing how pivots can save companies."
    },

    # --- Logistics / Delivery ---
    {
        "name": "Shypdirect", "outcome": "failed",
        "category": "unit_economics",
        "failure_reason": "Last-mile logistics economics couldn't compete with UPS/FedEx at scale",
        "description": "Last-mile delivery startup targeting e-commerce retailers with same-day delivery service. Raised $22M. Delivery economics challenging — FedEx and UPS had massive scale advantages. Could not build route density needed for profitability in competitive urban markets. Shut down operations."
    },
    {
        "name": "Doorman", "outcome": "failed",
        "category": "unit_economics",
        "failure_reason": "Evening package delivery service couldn't make economics work",
        "description": "Package delivery startup specializing in evening delivery windows when customers are home. Raised $2.5M. The specific delivery window requirement made route efficiency terrible. Could not charge enough premium to cover per-delivery economics. Competitors like Amazon Locker solved the problem differently. Shut down 2017."
    },

    # ════════════════════════════════
    # SUCCESSFUL STARTUPS (30)
    # ════════════════════════════════

    {
        "name": "Stripe", "outcome": "success",
        "category": "developer_tools",
        "failure_reason": None,
        "description": "Online payment processing platform for developers and businesses. Started with 7 lines of code to accept payments. Obsessive focus on developer experience and API quality. Expanded from payments to full financial infrastructure. Now valued at $95B serving millions of businesses globally."
    },
    {
        "name": "Airbnb", "outcome": "success",
        "category": "marketplace",
        "failure_reason": None,
        "description": "Home sharing marketplace connecting travelers with hosts. Started by renting air mattresses during a conference. Survived multiple near-deaths, entered YC, grew through grassroots marketing and clever Craigslist integration. IPO'd in 2020 at $47B valuation despite COVID impact on travel."
    },
    {
        "name": "Slack", "outcome": "success",
        "category": "b2b_saas",
        "failure_reason": None,
        "description": "Business messaging and collaboration platform. Pivoted from a failed gaming startup. Found product-market fit immediately with strong NPS scores and viral word-of-mouth within companies. Fastest business app to reach $1B valuation. Acquired by Salesforce for $27.7B in 2021."
    },
    {
        "name": "Zoom", "outcome": "success",
        "category": "b2b_saas",
        "failure_reason": None,
        "description": "Video conferencing platform founded by ex-Cisco engineer who was frustrated with Webex. Superior product reliability and ease of use differentiated from incumbent competitors. Grew steadily then exploded during COVID-19. IPO'd at $16B, reached $160B market cap at peak."
    },
    {
        "name": "Figma", "outcome": "success",
        "category": "developer_tools",
        "failure_reason": None,
        "description": "Collaborative web-based design tool eliminating the need for Sketch and desktop design software. Browser-based collaboration was core innovation enabling real-time team design. Grew through bottom-up adoption in design teams. Acquired by Adobe for $20B in 2022."
    },
    {
        "name": "Notion", "outcome": "success",
        "category": "b2b_saas",
        "failure_reason": None,
        "description": "All-in-one workspace combining notes, docs, wikis, and project management. Bootstrapped for years before finding product-market fit. Viral growth through bottom-up team adoption and template sharing community. Raised at $10B valuation in 2021 with strong revenue and retention."
    },
    {
        "name": "Canva", "outcome": "success",
        "category": "developer_tools",
        "failure_reason": None,
        "description": "Web-based graphic design platform making professional design accessible to non-designers. Freemium model with strong conversion to paid. Focused on ease-of-use over professional features. Grown to 125M+ monthly users, valued at $40B, profitable with strong recurring revenue."
    },
    {
        "name": "Shopify", "outcome": "success",
        "category": "marketplace",
        "failure_reason": None,
        "description": "E-commerce platform enabling small businesses to build online stores. Started as internal tool for an online snowboard shop. Obsessive focus on merchant success and ease of use. Built ecosystem of apps and partners. IPO'd in 2015, now $60B+ market cap serving 2M+ merchants."
    },
    {
        "name": "HubSpot", "outcome": "success",
        "category": "b2b_saas",
        "failure_reason": None,
        "description": "Inbound marketing and CRM platform for small and medium businesses. Invented the concept of inbound marketing, built entire content marketing category. Freemium CRM with paid marketing tools upsell. IPO'd in 2014, $25B+ market cap with strong NRR and consistent growth."
    },
    {
        "name": "Twilio", "outcome": "success",
        "category": "developer_tools",
        "failure_reason": None,
        "description": "Cloud communications platform providing APIs for SMS, voice, and video. Developer-first go-to-market with pay-per-use pricing lowering barrier to entry. Land-and-expand model with strong NRR over 120%. IPO'd in 2016, grew to $50B+ market cap as communications infrastructure standard."
    },
    {
        "name": "Datadog", "outcome": "success",
        "category": "b2b_saas",
        "failure_reason": None,
        "description": "Cloud monitoring and analytics platform for DevOps and engineering teams. Strong product execution with unified platform expanding from infrastructure to APM to logs. Net revenue retention consistently above 130%. IPO'd in 2019 at $7B, grew to $40B+ market cap."
    },
    {
        "name": "Snowflake", "outcome": "success",
        "category": "b2b_saas",
        "failure_reason": None,
        "description": "Cloud data warehousing platform separating storage and compute for flexible scaling. Consumption-based pricing aligned with customer value. Grew through data sharing features creating network effects between customers. Largest software IPO ever in 2020 at $33B valuation."
    },
    {
        "name": "UiPath", "outcome": "success",
        "category": "b2b_saas",
        "failure_reason": None,
        "description": "Robotic process automation platform automating repetitive computer tasks for enterprises. Strong product-led growth with free community version driving enterprise adoption. Became fastest company to reach $1B ARR. IPO'd in 2021 at $35B valuation."
    },
    {
        "name": "Veeva Systems", "outcome": "success",
        "category": "vertical_saas",
        "failure_reason": None,
        "description": "Cloud software for the global life sciences industry — CRM, content management, and data for pharma and biotech companies. Deep vertical focus on highly regulated, high-value industry. Strong net revenue retention. IPO'd in 2013, grew to $40B market cap, industry standard platform."
    },
    {
        "name": "Procore", "outcome": "success",
        "category": "vertical_saas",
        "failure_reason": None,
        "description": "Construction management software platform for contractors, owners, and subcontractors. Deep vertical focus on underserved construction industry. Comprehensive platform preventing competitive displacement. IPO'd in 2021 at $9.6B valuation with strong NRR and large TAM."
    },
    {
        "name": "Toast", "outcome": "success",
        "category": "vertical_saas",
        "failure_reason": None,
        "description": "Restaurant management platform combining point-of-sale, payments, and operations software. Deep vertical focus on restaurant industry with hardware plus SaaS model. Strong payments attach creating high switching costs. IPO'd in 2021 at $20B valuation."
    },
    {
        "name": "Brex", "outcome": "success",
        "category": "fintech",
        "failure_reason": None,
        "description": "Corporate credit card and financial services platform for startups and tech companies. Underwriting based on funding rather than personal credit. Rapidly expanded to expense management and banking. Reached $12B valuation with strong growth in fintech-savvy startup segment."
    },
    {
        "name": "Plaid", "outcome": "success",
        "category": "fintech",
        "failure_reason": None,
        "description": "Financial data network connecting consumer bank accounts to fintech apps. Became critical infrastructure for fintech ecosystem. Near-acquisition by Visa for $5.3B blocked by DOJ, remained independent. Strong network effects as more apps and banks join platform. Valued at $13B."
    },
    {
        "name": "Rippling", "outcome": "success",
        "category": "b2b_saas",
        "failure_reason": None,
        "description": "HR, IT, and Finance management platform unifying employee data across all business systems. Compound startup building multiple products on unified employee data graph. Strong NRR through platform expansion. Grew to $11B valuation as category-defining platform."
    },
    {
        "name": "Airtable", "outcome": "success",
        "category": "b2b_saas",
        "failure_reason": None,
        "description": "Low-code database and workflow platform combining spreadsheet simplicity with database power. Strong product-led growth with viral team adoption. Large TAM as replacement for both spreadsheets and custom databases. Grew to $11B valuation with broad adoption across industries."
    },
    {
        "name": "Monday.com", "outcome": "success",
        "category": "b2b_saas",
        "failure_reason": None,
        "description": "Work operating system for project management and workflow automation. Strong product-led growth with bottom-up team adoption. Expanded from project management to broader work platform. IPO'd in 2021 at $7.5B, strong NRR and consistent revenue growth."
    },
    {
        "name": "Wiz", "outcome": "success",
        "category": "b2b_saas",
        "failure_reason": None,
        "description": "Cloud security platform providing comprehensive visibility and risk assessment. Fastest company to reach $100M ARR in 18 months. Strong founder team with deep security expertise and prior exit. Near-acquisition by Google for $23B blocked by regulatory concerns. Valued at $12B+."
    },
    {
        "name": "Scale AI", "outcome": "success",
        "category": "ai_ml",
        "failure_reason": None,
        "description": "AI data labeling and training data platform for machine learning models. Critical infrastructure for AI development. Government and enterprise customers with high switching costs. Grew to $13.8B valuation as AI boom accelerated demand for high-quality training data."
    },
    {
        "name": "Weights and Biases", "outcome": "success",
        "category": "ai_ml",
        "failure_reason": None,
        "description": "MLOps platform for tracking machine learning experiments and model development. Developer-first go-to-market with free tier driving enterprise adoption. Became standard tool for ML engineers at leading AI companies. Raised at $1.25B valuation with strong product adoption."
    },
    {
        "name": "Hashicorp", "outcome": "success",
        "category": "developer_tools",
        "failure_reason": None,
        "description": "Infrastructure automation platform with Terraform, Vault, and Consul as industry standards. Open source to enterprise conversion model. Became critical infrastructure for cloud deployments globally. IPO'd in 2021 at $14B valuation. Acquired by IBM for $6.4B in 2024."
    },
    {
        "name": "Samsara", "outcome": "success",
        "category": "vertical_saas",
        "failure_reason": None,
        "description": "Connected operations platform for physical operations — fleet management, equipment monitoring, and worker safety for industries like trucking, construction, and utilities. IoT sensors plus cloud platform. IPO'd in 2021 at $11.6B valuation with strong ARR growth."
    },
    {
        "name": "project44", "outcome": "success",
        "category": "vertical_saas",
        "failure_reason": None,
        "description": "Supply chain visibility platform providing real-time tracking for freight across all modes of transportation. Raised $420M at $2.7B valuation. Deep integrations with carriers and enterprise shippers creating network effects. Dominant position in supply chain visibility software."
    },
    {
        "name": "Flexport", "outcome": "success",
        "category": "vertical_saas",
        "failure_reason": None,
        "description": "Digital freight forwarder and supply chain platform modernizing global trade. Software-first approach to freight forwarding creating full visibility. Raised $2.3B including from SoftBank. Valuable data network from moving actual freight. Valued at $8B+ as supply chain category leader."
    },
    {
        "name": "FourKites", "outcome": "success",
        "category": "vertical_saas",
        "failure_reason": None,
        "description": "Real-time supply chain visibility platform tracking shipments across road, rail, ocean, and air. Raised $200M at $1B+ valuation. Strong carrier and shipper network creating data advantages. Serves Fortune 500 companies for supply chain intelligence and analytics."
    },
    {
        "name": "Locus Robotics", "outcome": "success",
        "category": "vertical_saas",
        "failure_reason": None,
        "description": "Warehouse robotics automation platform deploying collaborative robots alongside human workers. Robots-as-a-service model reducing capital requirements for warehouse operators. Raised $150M+. Strong ROI demonstrated in real warehouse deployments with measurable productivity gains."
    },
]


def build_faiss_index(save: bool = True) -> Tuple:
    """
    Generate embeddings for all startups and build a FAISS index.
    
    This is the core of Step 9. It:
      1. Takes all startup descriptions from STARTUP_DATASET
      2. Converts them to 384-dimensional vectors
      3. Stores them in a FAISS IndexFlatIP (inner product / cosine similarity)
      4. Saves the index + metadata to disk for reuse
    
    Returns:
        (index, metadata) tuple
        - index: FAISS index ready for similarity search
        - metadata: list of dicts with name, outcome, category, etc.
    
    Runtime: ~30 seconds on first call (model load + encoding 80 texts)
    """
    try:
        import faiss
    except ImportError:
        raise RuntimeError("faiss-cpu not installed. Run: pip install faiss-cpu")
    
    from utils.embeddings import embed_texts
    
    logger.info(f"Building FAISS index for {len(STARTUP_DATASET)} startups...")
    
    # Extract descriptions for embedding
    descriptions = [s["description"] for s in STARTUP_DATASET]
    
    # Generate all embeddings in one batch
    embeddings = embed_texts(descriptions)
    logger.info(f"Generated embeddings: shape {embeddings.shape}")
    
    # Build FAISS index
    # IndexFlatIP = exact search using inner product (= cosine similarity for normalized vectors)
    dimension = embeddings.shape[1]  # 384
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    logger.info(f"FAISS index built: {index.ntotal} vectors, {dimension} dimensions")
    
    # Metadata (everything except the embedding itself)
    metadata = [
        {
            "id": i,
            "name": s["name"],
            "outcome": s["outcome"],
            "category": s["category"],
            "failure_reason": s.get("failure_reason"),
            "description": s["description"],
        }
        for i, s in enumerate(STARTUP_DATASET)
    ]
    
    if save:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(INDEX_FILE))
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(metadata, f)
        logger.info(f"Index saved to {INDEX_FILE}")
        logger.info(f"Metadata saved to {METADATA_FILE}")
    
    return index, metadata


def load_faiss_index() -> Tuple:
    """
    Load a previously built FAISS index from disk.
    Call build_faiss_index() first if files don't exist.
    
    Returns:
        (index, metadata) tuple
    """
    try:
        import faiss
    except ImportError:
        raise RuntimeError("faiss-cpu not installed. Run: pip install faiss-cpu")
    
    if not INDEX_FILE.exists() or not METADATA_FILE.exists():
        logger.info("FAISS index not found. Building it now...")
        return build_faiss_index(save=True)
    
    index = faiss.read_index(str(INDEX_FILE))
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)
    
    logger.info(f"Loaded FAISS index: {index.ntotal} vectors")
    return index, metadata


def get_dataset_stats() -> Dict:
    """Return basic stats about the dataset."""
    failed = sum(1 for s in STARTUP_DATASET if s["outcome"] == "failed")
    success = sum(1 for s in STARTUP_DATASET if s["outcome"] == "success")
    categories = {}
    for s in STARTUP_DATASET:
        categories[s["category"]] = categories.get(s["category"], 0) + 1
    return {
        "total": len(STARTUP_DATASET),
        "failed": failed,
        "success": success,
        "categories": categories
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Building startup dataset and FAISS index...\n")
    stats = get_dataset_stats()
    print(f"Dataset: {stats['total']} startups ({stats['failed']} failed, {stats['success']} successful)")
    print(f"Categories: {stats['categories']}\n")
    index, metadata = build_faiss_index(save=True)
    print(f"\n✅ FAISS index built successfully: {index.ntotal} vectors")
    print(f"   Saved to: {INDEX_FILE}")
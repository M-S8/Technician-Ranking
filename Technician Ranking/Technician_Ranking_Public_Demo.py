#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime
from IPython.display import display, HTML

# Set display options for better visibility during development
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 7000)
pd.set_option('max_colwidth', 50)

# ==================================================================================================
# SYNTHETIC DATA GENERATION
# ==================================================================================================
# This section replaces the original Snowflake data connection to demonstrate the model's functionality
# without exposing sensitive internal data.

def generate_synthetic_data():
    """
    Generates a synthetic dataset mimicking technician performance metrics.
    Includes: Job volume, Profit, Recalls, and categorical attributes.
    """
    print("Generating synthetic technician data...")
    
    # Configuration for synthetic generation
    num_techs = 500
    quarters = ['Quarter-1', 'Quarter-2', 'Quarter-3']
    
    # Lists for data generation
    districts = ['North-East', 'South-West', 'Mid-West', 'Pacific', 'Atlantic']
    job_titles = ['Service Tech I', 'Service Tech II', 'Senior Tech', 'Master Tech']
    
    records = []
    
    for q in quarters:
        for i in range(num_techs):
            tech_id = f"TECH_{i:03d}"
            
            # Randomly assign attributes
            district = np.random.choice(districts)
            title = np.random.choice(job_titles, p=[0.4, 0.3, 0.2, 0.1])
            
            # Generate Metrics
            # Volume: Poisson distribution to simulate job counts (skewed right)
            completes = np.random.poisson(lam=150)  # Average 150 jobs per quarter
            if completes == 0: completes = 1 # Avoid zero division
            
            # Profit: Normal distribution with some variance based on title
            base_profit = 120 + (job_titles.index(title) * 20) # Higher titles earn more
            avg_profit_per_job = np.random.normal(base_profit, 30)
            total_profit = completes * avg_profit_per_job
            
            # Recalls: Binomial distribution (Recalls happen with probability p)
            # Quality varies: Some techs are messy (high recall rate)
            recall_prob = np.random.beta(2, 50) # Beta distribution for realistic low probability (avg ~4%)
            parent_recalls = np.random.binomial(completes, recall_prob)
            
            records.append({
                'QUARTER_NO': q,
                'EMP_NPSID': tech_id,
                'JOBTITLE': title,
                'DISTRICT_NAME': district,
                'COMPLETES': completes,
                'ESTIM_PROFIT': total_profit,
                'PARENT_RECALLS': parent_recalls
            })
            
    df = pd.DataFrame(records)
    return df

def generate_company_financials(tech_df):
    """
    Aggregates the synthetic tech data to create company-level benchmarks.
    """
    print("Generating company financial benchmarks...")
    fin_df = tech_df.groupby('QUARTER_NO').agg(
        COMPLETES=('COMPLETES', 'sum'),
        REVENUE=('ESTIM_PROFIT', lambda x: x.sum() * 1.2), # Assume 20% margin for revenue simulation
        PROFIT=('ESTIM_PROFIT', 'sum')
    ).reset_index()
    return fin_df

# Load Data
raw_df = generate_synthetic_data()
fin_df = generate_company_financials(raw_df)

print("\nData Generation Complete.")
print(f"Technician Records: {len(raw_df)}")
print(f"Company Financials: {len(fin_df)}")

# ==================================================================================================
# ANALYTICAL MODEL: QUALITY ADJUSTED BAYESIAN RANKING
# ==================================================================================================

def calculate_bayesian_score(df, company_df=None):
    """
    Calculates a Bayesian Average Score for technician profit, adjusted for Quality (Recall Rate).
    
    1. Profit Score = (v / (v + m)) * R + (m / (v + m)) * C
    2. Quality Penalty = (1 - Recall Rate)
    3. Final Score = Profit Score * Quality Penalty
    
    where:
    R = Average Profit per Job for the technician
    v = Number of completes for the technician
    m = Minimum completes required to be listed (median completes of the quarter)
    C = The mean profit per job across the whole company for that quarter
    """
    scored_df = df.copy()
    
    # Normalize columns to uppercase (just in case)
    scored_df.columns = [c.upper() for c in scored_df.columns]
    
    if company_df is not None:
        company_df_norm = company_df.copy()
        company_df_norm.columns = [c.upper() for c in company_df_norm.columns]
    
    # Calculate Average Profit per Job for each tech
    scored_df['AVG_PROFIT_PER_JOB'] = scored_df.apply(
        lambda row: row['ESTIM_PROFIT'] / row['COMPLETES'] if row['COMPLETES'] > 0 else 0, axis=1
    )
    
    # Calculate Recall Rate (Quality Metric)
    # Parent Recalls / Completes
    scored_df['RECALL_RATE'] = scored_df.apply(
        lambda row: (row['PARENT_RECALLS'] / row['COMPLETES']) if row['COMPLETES'] > 0 else 0.0, axis=1
    )
    
    # Create a list to store results
    results = []
    
    quarters = scored_df['QUARTER_NO'].unique()
    
    for quarter in sorted(quarters):
        q_data = scored_df[scored_df['QUARTER_NO'] == quarter].copy()
        
        # C: Mean profit per job for the entire quarter (Prior Mean)
        if company_df is not None and not company_df_norm[company_df_norm['QUARTER_NO'] == quarter].empty:
            c_data = company_df_norm[company_df_norm['QUARTER_NO'] == quarter].iloc[0]
            C = c_data['PROFIT'] / c_data['COMPLETES'] if c_data['COMPLETES'] > 0 else 0
        else:
            total_profit = q_data['ESTIM_PROFIT'].sum()
            total_completes = q_data['COMPLETES'].sum()
            C = total_profit / total_completes if total_completes > 0 else 0
        
        # m: Smoothing factor (Median jobs per tech)
        # This controls how much we shrink towards the mean.
        m = q_data['COMPLETES'].quantile(0.50)
        
        # Apply Weighted Rating Formula
        def weighted_rating(x, m=m, C=C):
            v = x['COMPLETES']
            R = x['AVG_PROFIT_PER_JOB']
            return (v / (v + m) * R) + (m / (v + m) * C)
            
        q_data['BAYESIAN_SCORE'] = q_data.apply(weighted_rating, axis=1)
        
        # Apply Quality Penalty
        # Final Score = Bayesian Profit Score * (1 - Recall Rate)
        # If a tech has 10% recall rate, their profit score is discounted by 10%
        q_data['QUALITY_ADJUSTED_SCORE'] = q_data['BAYESIAN_SCORE'] * (1 - q_data['RECALL_RATE'])
        
        q_data['QUARTER_GLOBAL_AVG'] = C
        q_data['SMOOTHING_FACTOR_M'] = m
        
        results.append(q_data)
        
    return pd.concat(results)

def display_bayesian_ranking(df, company_df=None):
    """
    Prints the Top and Bottom performers to the console.
    """
    ranked_df = calculate_bayesian_score(df, company_df)
    
    cols = ['RANK', 'EMP_NPSID', 'JOBTITLE', 'DISTRICT_NAME', 'COMPLETES', 'RECALL_RATE', 'ESTIM_PROFIT', 'AVG_PROFIT_PER_JOB', 'QUALITY_ADJUSTED_SCORE']
    cols = [c for c in cols if c in ranked_df.columns]
    
    quarters = ranked_df['QUARTER_NO'].unique()
    
    for quarter in sorted(quarters):
        print(f"\n{'='*100}")
        print(f"QUARTER: {quarter} | Global Avg Profit/Job: {ranked_df[ranked_df['QUARTER_NO']==quarter]['QUARTER_GLOBAL_AVG'].iloc[0]:.2f}")
        print(f"{'='*100}")
        
        q_data = ranked_df[ranked_df['QUARTER_NO'] == quarter].copy()
        # Sort by the new Quality Adjusted Score
        q_data.sort_values('QUALITY_ADJUSTED_SCORE', ascending=False, inplace=True)
        
        # Assign Rank
        q_data['RANK'] = range(1, len(q_data) + 1)
        
        print(f"\n--- TOP 10 TECHNICIANS (Quality Adjusted Profit Score) ---")
        print(q_data.head(10)[cols].to_string(index=False))
        
        print(f"\n--- BOTTOM 10 TECHNICIANS (Quality Adjusted Profit Score) ---")
        print(q_data.tail(10)[cols].to_string(index=False))

# ==================================================================================================
# REPORT GENERATION
# ==================================================================================================

def generate_executive_html_report(df, company_df=None, filename="Technician_Performance_Report_Public.html"):
    """
    Generates an HTML executive summary report for technician rankings.
    """
    print(f"\nGenerating Executive Summary Report: {filename}")
    
    ranked_df = calculate_bayesian_score(df, company_df)
    
    html_content = """
    <html>
    <head>
        <title>Technician Performance Executive Summary</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; color: #333; background-color: #f4f7f6; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-radius: 8px; }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 15px; margin-bottom: 30px; }
            h2 { color: #2c3e50; margin-top: 40px; border-left: 5px solid #3498db; padding-left: 10px; }
            h3 { color: #7f8c8d; margin-top: 25px; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 14px; }
            th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }
            th { background-color: #34495e; color: white; text-transform: uppercase; letter-spacing: 0.5px; font-size: 12px; }
            tr:nth-child(even) { background-color: #f8f9fa; }
            tr:hover { background-color: #e9ecef; }
            .explanation { background-color: #e8f6f3; padding: 20px; border-radius: 5px; border-left: 5px solid #1abc9c; margin-bottom: 30px; }
            .metric-card { background: #fff; border: 1px solid #ddd; padding: 15px; border-radius: 5px; display: inline-block; margin-right: 20px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
            .metric-label { font-size: 12px; color: #7f8c8d; text-transform: uppercase; display: block; margin-bottom: 5px; }
            .metric-value { font-size: 20px; font-weight: bold; color: #2c3e50; }
            .badge { padding: 3px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; color: white; }
            .badge-top { background-color: #27ae60; }
            .badge-bottom { background-color: #e74c3c; }
            .footer { margin-top: 50px; font-size: 12px; color: #aaa; text-align: center; border-top: 1px solid #eee; padding-top: 20px; }
        </style>
    </head>
    <body>
    <div class="container">
        <h1>Quarterly Technician Performance Executive Summary</h1>
        
        <div class="explanation">
            <h3 style="margin-top:0; color: #16a085;">Methodology: Quality Adjusted Bayesian Ranking</h3>
            <p>This report ranks technicians based on a <strong>Bayesian Weighted Average</strong> of their profit per job, penalized by their <strong>Recall Rate</strong>.</p>
            <p><strong>Why this method?</strong></p>
            <ul>
                <li>Standard averages can be misleading for technicians with low job volume (volatility).</li>
                <li>This method "shrinks" the average profit of technicians with low volume towards the <strong>Company Global Average</strong>.</li>
                <li><strong>New:</strong> Scores are now discounted by the technician's Recall Rate. A 10% recall rate reduces the profit score by 10%.</li>
            </ul>
            <p style="font-family: monospace; background: rgba(255,255,255,0.5); padding: 10px; border-radius: 4px;">
                <strong>Score Formula:</strong> S = [(v / (v + m)) * R + (m / (v + m)) * C] * (1 - RecallRate)
            </p>
            <ul style="font-size: 0.9em; color: #555;">
                <li><strong>v:</strong> Technician's job count (Evidence)</li>
                <li><strong>R:</strong> Technician's average profit (Observed Mean)</li>
                <li><strong>C:</strong> Company-wide average profit per job (Prior Mean / Benchmark)</li>
                <li><strong>m:</strong> Smoothing factor (Median job count of the quarter)</li>
                <li><strong>RecallRate:</strong> % of jobs resulting in a repeat call (Penalty)</li>
            </ul>
        </div>
    """
    
    cols = ['RANK', 'EMP_NPSID', 'JOBTITLE', 'DISTRICT_NAME', 'COMPLETES', 'RECALL_RATE', 'ESTIM_PROFIT', 'AVG_PROFIT_PER_JOB', 'QUALITY_ADJUSTED_SCORE']
    display_cols = ['Rank', 'Tech ID', 'Job Title', 'District', 'Completes', 'Recall Rate', 'Total Profit ($)', 'Avg Profit/Job ($)', 'Quality Score']
    
    quarters = ranked_df['QUARTER_NO'].unique()
    
    for quarter in sorted(quarters):
        q_data = ranked_df[ranked_df['QUARTER_NO'] == quarter].copy()
        q_data.sort_values('QUALITY_ADJUSTED_SCORE', ascending=False, inplace=True)
        q_data['RANK'] = range(1, len(q_data) + 1)
        
        global_avg = q_data['QUARTER_GLOBAL_AVG'].iloc[0] if 'QUARTER_GLOBAL_AVG' in q_data.columns else 0
        median_jobs = q_data['SMOOTHING_FACTOR_M'].iloc[0] if 'SMOOTHING_FACTOR_M' in q_data.columns else 0
        
        format_currency = lambda x: f"{x:,.2f}"
        format_percent = lambda x: f"{x*100:.1f}%"
        
        q_data['ESTIM_PROFIT'] = q_data['ESTIM_PROFIT'].apply(format_currency)
        q_data['AVG_PROFIT_PER_JOB'] = q_data['AVG_PROFIT_PER_JOB'].apply(format_currency)
        q_data['QUALITY_ADJUSTED_SCORE'] = q_data['QUALITY_ADJUSTED_SCORE'].apply(format_currency)
        q_data['RECALL_RATE'] = q_data['RECALL_RATE'].apply(format_percent)
        
        top_10 = q_data.head(10)[cols]
        top_10.columns = display_cols
        
        bottom_10 = q_data.tail(10)[cols]
        bottom_10.columns = display_cols
        
        html_content += f"""
        <hr style="border: 0; border-top: 1px solid #eee; margin: 40px 0;">
        <h2>{quarter} Performance Overview</h2>
        
        <div style="margin-bottom: 20px;">
            <div class="metric-card">
                <span class="metric-label">Company Avg Profit/Job</span>
                <span class="metric-value">${global_avg:,.2f}</span>
            </div>
            <div class="metric-card">
                <span class="metric-label">Median Job Volume</span>
                <span class="metric-value">{int(median_jobs)}</span>
            </div>
             <div class="metric-card">
                <span class="metric-label">Total Technicians</span>
                <span class="metric-value">{len(q_data)}</span>
            </div>
        </div>

        <h3><span class="badge badge-top">TOP 10</span> Highest Performing Technicians</h3>
        {top_10.to_html(index=False, classes='table', border=0)}
        
        <h3><span class="badge badge-bottom">BOTTOM 10</span> Lowest Performing Technicians</h3>
        {bottom_10.to_html(index=False, classes='table', border=0)}
        """
        
    html_content += f"""
        <div class="footer">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Technician Ranking Model Public Demo
        </div>
    </div>
    </body>
    </html>
    """
    
    with open(filename, "w", encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML Report generated successfully: {os.path.abspath(filename)}")

# ==================================================================================================
# EXECUTION
# ==================================================================================================

if __name__ == "__main__":
    print("Starting Public Demo Script...")
    
    # 1. Display Ranking in Console
    display_bayesian_ranking(raw_df, fin_df)
    
    # 2. Generate HTML Report
    generate_executive_html_report(raw_df, fin_df)
    
    print("\nDemo Completed Successfully.")

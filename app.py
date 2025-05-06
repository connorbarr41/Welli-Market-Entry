import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(page_title="LatAm Financial Model", layout="wide")

def fetch_exchange_rates(base_currency='USD'):
    currencies = ['PEN', 'COP', 'CLP']
    rates = {}
    
    for currency in currencies:
        try:
            response = requests.get(f'https://api.exchangerate-api.com/v4/latest/{base_currency}')
            data = response.json()
            rates[currency] = data['rates'].get(currency, None)
        except:
            rates[currency] = None
    
    return rates

def calculate_metrics(inputs):
    annual_patients = inputs['monthly_patients'] * 12
    total_procedures = annual_patients * inputs['procedure_cost']
    financed_amount = total_procedures * (inputs['financing_rate'] / 100)
    
    revenue = {
        'Interest': financed_amount * (inputs['interest_rate'] / 100),
        'Medical Discount': financed_amount * (inputs['medical_discount'] / 100),
        'Insurance': financed_amount * (inputs['insurance_commission'] / 100)
    }
    
    costs = {
        'Funding': financed_amount * (inputs['funding_cost'] / 100),
        'Operating': annual_patients * inputs['operating_cost'],
        'Bad Debt': financed_amount * (inputs['bad_debt'] / 100),
        'Compliance': inputs['compliance_cost']
    }
    
    total_revenue = sum(revenue.values())
    total_costs = sum(costs.values())
    
    pre_tax_profit = total_revenue - total_costs
    tax = pre_tax_profit * (inputs['corporate_tax'] / 100)
    net_profit = pre_tax_profit - tax
    usd_profit = net_profit / inputs['exchange_rate']
    
    return {
        'Annual Patients': annual_patients,
        'Total Revenue': total_revenue,
        'Total Costs': total_costs,
        'Pre-tax Profit': pre_tax_profit,
        'Tax': tax,
        'Net Profit': net_profit,
        'USD Net Profit': usd_profit,
        'Revenue Breakdown': revenue,
        'Cost Breakdown': costs
    }

def create_waterfall_chart(results):
    fig = go.Figure(go.Waterfall(
        name="Financial Breakdown",
        orientation="v",
        measure=["relative", "relative", "relative", "total", 
                "relative", "relative", "relative", "relative", "total",
                "relative", "total"],
        x=["Interest", "Medical Discount", "Insurance", "Total Revenue",
           "Funding", "Operating", "Bad Debt", "Compliance", "Total Costs",
           "Tax", "Net Profit"],
        y=[results['Revenue Breakdown']['Interest'],
           results['Revenue Breakdown']['Medical Discount'],
           results['Revenue Breakdown']['Insurance'],
           results['Total Revenue'],
           -results['Cost Breakdown']['Funding'],
           -results['Cost Breakdown']['Operating'],
           -results['Cost Breakdown']['Bad Debt'],
           -results['Cost Breakdown']['Compliance'],
           -results['Total Costs'],
           -results['Tax'],
           results['Net Profit']],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(title='Financial Breakdown', height=600)
    return fig

# Initialize session state
if 'current_inputs' not in st.session_state:
    st.session_state.current_inputs = {
        'country': 'Peru',
        'monthly_patients': 100.0,
        'procedure_cost': 5000.0,
        'financing_rate': 70.0,
        'interest_rate': 15.0,
        'medical_discount': 10.0,
        'insurance_commission': 5.0,
        'funding_cost': 8.0,
        'operating_cost': 200.0,
        'bad_debt': 5.0,
        'compliance_cost': 50000.0,
        'corporate_tax': 30.0,
        'exchange_rate': 3.75,
        'inflation_rate': 4.0,
        'patient_growth': 5.0
    }

# App title and description
st.title('LatAm Financial Model')
st.markdown('Financial modeling tool for medical procedures in Latin America')

# Sidebar for inputs
st.sidebar.header('Model Parameters')

# Country selection
country = st.sidebar.selectbox(
    'Country',
    ['Peru', 'Colombia', 'Chile'],
    index=['Peru', 'Colombia', 'Chile'].index(st.session_state.current_inputs['country'])
)

# Fetch current exchange rates
exchange_rates = fetch_exchange_rates()
default_rates = {'PEN': 3.75, 'COP': 4000, 'CLP': 850}
country_currency = {'Peru': 'PEN', 'Colombia': 'COP', 'Chile': 'CLP'}
current_rate = exchange_rates.get(country_currency[country], default_rates[country_currency[country]])

# Input parameters
col1, col2 = st.sidebar.columns(2)

with col1:
    monthly_patients = st.number_input('Monthly Patients', 
                                     value=st.session_state.current_inputs['monthly_patients'],
                                     min_value=0.0)
    procedure_cost = st.number_input('Procedure Cost',
                                   value=st.session_state.current_inputs['procedure_cost'],
                                   min_value=0.0)
    financing_rate = st.number_input('Financing Rate (%)',
                                   value=st.session_state.current_inputs['financing_rate'],
                                   min_value=0.0, max_value=100.0)
    interest_rate = st.number_input('Interest Rate (%)',
                                  value=st.session_state.current_inputs['interest_rate'],
                                  min_value=0.0, max_value=100.0)
    medical_discount = st.number_input('Medical Discount (%)',
                                     value=st.session_state.current_inputs['medical_discount'],
                                     min_value=0.0, max_value=100.0)
    insurance_commission = st.number_input('Insurance Commission (%)',
                                        value=st.session_state.current_inputs['insurance_commission'],
                                        min_value=0.0, max_value=100.0)
    funding_cost = st.number_input('Funding Cost (%)',
                                 value=st.session_state.current_inputs['funding_cost'],
                                 min_value=0.0, max_value=100.0)

with col2:
    operating_cost = st.number_input('Operating Cost per Patient',
                                   value=st.session_state.current_inputs['operating_cost'],
                                   min_value=0.0)
    bad_debt = st.number_input('Bad Debt (%)',
                             value=st.session_state.current_inputs['bad_debt'],
                             min_value=0.0, max_value=100.0)
    compliance_cost = st.number_input('Annual Compliance Cost',
                                    value=st.session_state.current_inputs['compliance_cost'],
                                    min_value=0.0)
    corporate_tax = st.number_input('Corporate Tax Rate (%)',
                                  value=st.session_state.current_inputs['corporate_tax'],
                                  min_value=0.0, max_value=100.0)
    exchange_rate = st.number_input('Exchange Rate',
                                  value=current_rate,
                                  min_value=0.0)
    inflation_rate = st.number_input('Inflation Rate (%)',
                                   value=st.session_state.current_inputs['inflation_rate'],
                                   min_value=0.0, max_value=100.0)
    patient_growth = st.number_input('Patient Growth Rate (%)',
                                   value=st.session_state.current_inputs['patient_growth'],
                                   min_value=0.0, max_value=100.0)

# Update session state
current_inputs = {
    'country': country,
    'monthly_patients': monthly_patients,
    'procedure_cost': procedure_cost,
    'financing_rate': financing_rate,
    'interest_rate': interest_rate,
    'medical_discount': medical_discount,
    'insurance_commission': insurance_commission,
    'funding_cost': funding_cost,
    'operating_cost': operating_cost,
    'bad_debt': bad_debt,
    'compliance_cost': compliance_cost,
    'corporate_tax': corporate_tax,
    'exchange_rate': exchange_rate,
    'inflation_rate': inflation_rate,
    'patient_growth': patient_growth
}

st.session_state.current_inputs = current_inputs

# Calculate results
results = calculate_metrics(current_inputs)

# Display results
st.header('Financial Analysis Results')

col1, col2, col3 = st.columns(3)

with col1:
    st.metric('Annual Patients', f"{int(results['Annual Patients']):,}")
    st.metric('Total Revenue', f"${results['Total Revenue']:,.2f}")
    st.metric('Total Costs', f"${results['Total Costs']:,.2f}")

with col2:
    st.metric('Pre-tax Profit', f"${results['Pre-tax Profit']:,.2f}")
    st.metric('Tax', f"${results['Tax']:,.2f}")
    st.metric('Net Profit', f"${results['Net Profit']:,.2f}")

with col3:
    st.metric('USD Net Profit', f"${results['USD Net Profit']:,.2f}")
    st.metric('Revenue per Patient', f"${results['Total Revenue']/results['Annual Patients']:,.2f}")
    st.metric('Cost per Patient', f"${results['Total Costs']/results['Annual Patients']:,.2f}")

# Display waterfall chart
st.plotly_chart(create_waterfall_chart(results), use_container_width=True)

# Forecast section
st.header('5-Year Forecast')

forecast_years = 5
forecast_data = []

for year in range(forecast_years):
    year_inputs = current_inputs.copy()
    year_inputs['monthly_patients'] *= (1 + patient_growth/100) ** year
    year_inputs['procedure_cost'] *= (1 + inflation_rate/100) ** year
    year_inputs['operating_cost'] *= (1 + inflation_rate/100) ** year
    year_inputs['compliance_cost'] *= (1 + inflation_rate/100) ** year
    
    year_results = calculate_metrics(year_inputs)
    forecast_data.append({
        'Year': f'Year {year + 1}',
        'Patients': year_results['Annual Patients'],
        'Revenue': year_results['Total Revenue'],
        'Costs': year_results['Total Costs'],
        'Net Profit': year_results['Net Profit'],
        'USD Profit': year_results['USD Net Profit']
    })

forecast_df = pd.DataFrame(forecast_data)
st.dataframe(forecast_df.style.format({
    'Patients': '{:,.0f}',
    'Revenue': '${:,.2f}',
    'Costs': '${:,.2f}',
    'Net Profit': '${:,.2f}',
    'USD Profit': '${:,.2f}'
}))

# Create forecast chart
fig = go.Figure()

metrics = ['Revenue', 'Costs', 'Net Profit', 'USD Profit']
for metric in metrics:
    fig.add_trace(go.Scatter(
        name=metric,
        x=forecast_df['Year'],
        y=forecast_df[metric],
        mode='lines+markers'
    ))

fig.update_layout(
    title='5-Year Forecast',
    xaxis_title='Year',
    yaxis_title='Amount ($)',
    height=600
)

st.plotly_chart(fig, use_container_width=True)

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
        'monthly_patients': 93100.0,
        'procedure_cost': 1200.0,
        'financing_rate': 100.0,
        'interest_rate': 28.0,
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
# enforce 100% financing everywhere
    st.session_state.current_inputs['financing_rate'] = 100.0

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
                                     value=float(st.session_state.current_inputs['monthly_patients']),
                                     min_value=0.0)
    procedure_cost = st.number_input('Procedure Cost',
                                   value=float(st.session_state.current_inputs['procedure_cost']),
                                   min_value=0.0)
    interest_rate = st.number_input('Interest Rate (%)',
                                  value=float(st.session_state.current_inputs['interest_rate']),
                                  min_value=0.0, max_value=100.0)
    medical_discount = st.number_input('Medical Discount (%)',
                                     value=float(st.session_state.current_inputs['medical_discount']),
                                     min_value=0.0, max_value=100.0)
    insurance_commission = st.number_input('Insurance Commission (%)',
                                        value=float(st.session_state.current_inputs['insurance_commission']),
                                        min_value=0.0, max_value=100.0)
    funding_cost = st.number_input('Funding Cost (%)',
                                 value=float(st.session_state.current_inputs['funding_cost']),
                                 min_value=0.0, max_value=100.0)

with col2:
    operating_cost = st.number_input('Operating Cost per Patient',
                                   value=float(st.session_state.current_inputs['operating_cost']),
                                   min_value=0.0)
    bad_debt = st.number_input('Bad Debt (%)',
                             value=float(st.session_state.current_inputs['bad_debt']),
                             min_value=0.0, max_value=100.0)
    compliance_cost = st.number_input('Annual Compliance Cost',
                                    value=float(st.session_state.current_inputs['compliance_cost']),
                                    min_value=0.0)
    corporate_tax = st.number_input('Corporate Tax Rate (%)',
                                  value=float(st.session_state.current_inputs['corporate_tax']),
                                  min_value=0.0, max_value=100.0)
    exchange_rate = st.number_input('Exchange Rate',
                                  value=current_rate,
                                  min_value=0.0)
    inflation_rate = st.number_input('Inflation Rate (%)',
                                   value=float(st.session_state.current_inputs['inflation_rate']),
                                   min_value=0.0, max_value=100.0)
    patient_growth = st.number_input('Patient Growth Rate (%)',
                                   value=float(st.session_state.current_inputs['patient_growth']),
                                   min_value=0.0, max_value=100.0)

# Update session state
current_inputs = {
    'country': country,
    'monthly_patients': monthly_patients,
    'procedure_cost': procedure_cost,
    'financing_rate': 100.0,
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
# ---------------------------------------
# Sensitivity Analysis
# ---------------------------------------
st.header("Sensitivity Analysis")

st.subheader("Adjust Variables")

col1, col2, col3 = st.columns(3)

with col1:
    sens_interest = st.slider("Interest Rate (%)", 5.0, 90.0, float(current_inputs['interest_rate']), 0.5)

with col2:
    sens_bad_debt = st.slider("Bad Debt (%)", 0.0, 25.0, float(current_inputs['bad_debt']), 0.5)

with col3:
    sens_revenue_mult = st.slider("Revenue Multiplier", 0.8, 1.2, 1.0, 0.01)

# Apply sensitivity inputs
sensitivity_inputs = current_inputs.copy()
sensitivity_inputs['interest_rate'] = sens_interest
sensitivity_inputs['bad_debt'] = sens_bad_debt
for key in ['procedure_cost']:
    sensitivity_inputs[key] *= sens_revenue_mult

sensitivity_results = calculate_metrics(sensitivity_inputs)

st.metric("Adjusted Net Profit", f"${sensitivity_results['Net Profit']:,.2f}")
st.metric("Adjusted Total Revenue", f"${sensitivity_results['Total Revenue']:,.2f}")
st.metric("Adjusted Total Costs", f"${sensitivity_results['Total Costs']:,.2f}")
# ---------------------------------------
# Monte Carlo Simulation
# ---------------------------------------
st.header("Monte Carlo Simulation")
redo_monte = st.button("Redo Monte Carlo")
st.write("Simulating 1,000 scenarios with widened variability—click the button to rerun.")

current_inputs = st.session_state.current_inputs
current_inputs['financing_rate'] = 100.0
simulations   = 1000

# run on first load or when button is clicked
if 'monte_df' not in st.session_state or redo_monte:
    np.random.seed(None)
    results = []

    for _ in range(simulations):
        sim = current_inputs.copy()

        # revenue drivers
        sim['monthly_patients'] = max(
            0,
            np.random.normal(
                current_inputs['monthly_patients'],
                current_inputs['monthly_patients'] * 0.2
            )
        )
        sim['procedure_cost'] *= np.random.uniform(0.7, 1.3)

        # cost drivers
        sim['operating_cost']  *= np.random.uniform(0.7, 1.3)
        sim['compliance_cost'] *= np.random.uniform(0.8, 1.2)
        sim['funding_cost']    *= np.random.uniform(0.7, 1.3)

        # fees & discounts
        sim['medical_discount']     *= np.random.uniform(0.7, 1.3)
        sim['insurance_commission'] *= np.random.uniform(0.7, 1.3)

        # macro & tax variables
        sim['interest_rate']  = np.random.uniform(5.0, 90.0)
        sim['bad_debt']       = np.random.uniform(1.0, 25.0)
        sim['exchange_rate'] *= np.random.uniform(0.9, 1.1)
        sim['inflation_rate'] = np.random.uniform(
            current_inputs['inflation_rate'] * 0.8,
            current_inputs['inflation_rate'] * 1.2
        )
        sim['corporate_tax'] = np.random.uniform(
            current_inputs['corporate_tax'] * 0.8,
            current_inputs['corporate_tax'] * 1.2
        )

        # growth assumption
        sim['patient_growth'] = np.random.uniform(
            current_inputs['patient_growth'] * 0.1,
            current_inputs['patient_growth'] * 1.5
        )

        results.append(calculate_metrics(sim)['Net Profit'])

    st.session_state.monte_df = pd.DataFrame(results, columns=["Net Profit"])


# pull in the DataFrame
monte_df = st.session_state.monte_df

# chart & stats
fig = go.Figure(data=[go.Histogram(x=monte_df["Net Profit"], nbinsx=50)])
fig.update_layout(
    title="Monte Carlo Simulation: Net Profit Distribution",
    xaxis_title="Net Profit",
    yaxis_title="Frequency",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Summary Statistics")
st.write(f"Mean Net Profit: ${monte_df['Net Profit'].mean():,.2f}")
st.write(f"Probability of Loss: {(monte_df['Net Profit'] < 0).mean()*100:.1f}%")

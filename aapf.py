
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
import io
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Battery Health Prediction - CEF Analysis",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 Battery Health Prediction - CEF Analysis")
st.markdown("Upload your battery cycler data to calculate CEF (Capacity Estimation Filter) and extract health features")

# Sidebar for options
st.sidebar.header("Processing Options")
remove_first_row = st.sidebar.checkbox("Remove First Row (Conditioning Cycle)", value=True, 
                                      help="Remove the first cycle which is typically a conditioning cycle with outlier values")

# File upload
uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'])

if uploaded_file is not None:
    try:
        # Read Excel file
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_name = excel_file.sheet_names[0]
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)

        st.success(f"File uploaded successfully! Sheet: {sheet_name}")

        # Show original data preview
        st.subheader("📊 Original Data Preview")
        st.write(f"Dataset shape: {df.shape}")
        st.dataframe(df.head())

        # Processing functions
        def time_to_decimal_hours(time_str):
            time_obj = datetime.strptime(str(time_str), '%H:%M:%S.%f')
            decimal_hours = time_obj.hour + time_obj.minute/60 + time_obj.second/3600 + time_obj.microsecond/3600000000
            return decimal_hours

        # Data processing
        with st.spinner("Processing data..."):
            # Convert time and clean data
            df['Time_Hours'] = df['Time'].apply(time_to_decimal_hours)
            df_cleaned = df.drop(['Time', 'Date'], axis=1)
            columns_order = ['Sr. No.', 'Time_Hours', 'Voltage (mV)', 'Current (mA)', 'Capacity (mAh)', 'Energy (mWh)']
            df_final = df_cleaned[columns_order]
            df_final = df_final[df_final['Current (mA)'] != 0]
            df_final = df_final.reset_index(drop=True)

            # Find phase transitions
            df_final['Current_Sign'] = df_final['Current (mA)'] > 0
            df_final['Sign_Change'] = df_final['Current_Sign'] != df_final['Current_Sign'].shift(1)

            end_of_phases = []
            for i in range(1, len(df_final)):
                if df_final.iloc[i]['Sign_Change']:
                    end_of_phases.append(i-1)

            if df_final.iloc[-1]['Current (mA)'] < 0:
                end_of_phases.append(len(df_final) - 1)

            # Create final dataset
            final_dataset = df_final.iloc[end_of_phases].copy()
            final_dataset = final_dataset.reset_index(drop=True)
            final_dataset['Sr. No.'] = range(1, len(final_dataset) + 1)
            final_dataset = final_dataset.drop(['Current_Sign', 'Sign_Change'], axis=1)

            # Add capacity/energy columns
            final_dataset['Charge_Capacity'] = final_dataset.apply(
                lambda row: row['Capacity (mAh)'] if row['Current (mA)'] > 0 else 0, axis=1
            )
            final_dataset['Discharge_Capacity'] = final_dataset.apply(
                lambda row: row['Capacity (mAh)'] if row['Current (mA)'] < 0 else 0, axis=1
            )
            final_dataset['Charge_Energy'] = final_dataset.apply(
                lambda row: row['Energy (mWh)'] if row['Current (mA)'] > 0 else 0, axis=1
            )
            final_dataset['Discharge_Energy'] = final_dataset.apply(
                lambda row: row['Energy (mWh)'] if row['Current (mA)'] < 0 else 0, axis=1
            )

            # Shift discharge columns
            final_dataset['Discharge_Capacity'] = final_dataset['Discharge_Capacity'].shift(-1).fillna(0)
            final_dataset['Discharge_Energy'] = final_dataset['Discharge_Energy'].shift(-1).fillna(0)

            # Remove zero rows
            final_dataset = final_dataset[
                (final_dataset['Charge_Capacity'] > 0) & 
                (final_dataset['Discharge_Capacity'] > 0) & 
                (final_dataset['Charge_Energy'] > 0) & 
                (final_dataset['Discharge_Energy'] > 0)
            ]
            final_dataset = final_dataset.reset_index(drop=True)
            final_dataset['Sr. No.'] = range(1, len(final_dataset) + 1)
            final_dataset.insert(1, 'Cycle_Number', range(1, len(final_dataset) + 1))

            # Calculate efficiencies
            final_dataset['Coulombic_Efficiency'] = final_dataset['Discharge_Capacity'] / final_dataset['Charge_Capacity']
            final_dataset['Energy_Efficiency'] = final_dataset['Discharge_Energy'] / final_dataset['Charge_Energy']

            # Calculate CEF
            CE = final_dataset['Coulombic_Efficiency']
            EE = final_dataset['Energy_Efficiency']
            CEF = 2 / (1 / np.exp(-10 * (1 - CE)) + 1 / np.exp(-10 * (1 - EE)))
            final_dataset['CEF'] = CEF

            # Optional: Remove first row
            if remove_first_row:
                original_size = len(final_dataset)
                final_dataset = final_dataset.iloc[1:].copy()
                final_dataset = final_dataset.reset_index(drop=True)
                final_dataset['Sr. No.'] = range(1, len(final_dataset) + 1)
                final_dataset['Cycle_Number'] = range(1, len(final_dataset) + 1)
                st.info(f"First row (conditioning cycle) removed. Dataset size: {original_size} → {len(final_dataset)} cycles")

        st.success("✅ Data processing completed!")

        # Show processed data
        st.subheader("🔧 Processed Data")
        st.write(f"Final dataset shape: {final_dataset.shape}")
        st.dataframe(final_dataset.head(10))

        # CEF Statistics
        st.subheader("📈 CEF Analysis - First 10 Cycles")

        first_10_cycles = final_dataset.head(10).copy()
        cef_values = first_10_cycles['CEF'].values
        cycle_numbers = first_10_cycles['Cycle_Number'].values

        # Calculate statistics
        X = cycle_numbers.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, cef_values)
        cef_slope = model.coef_[0]
        cef_range = cef_values.max() - cef_values.min()
        cef_std = np.std(cef_values, ddof=1)
        cef_mean = np.mean(cef_values)

        # Display statistics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("CEF Slope", f"{cef_slope:.6f}", help="Linear regression slope")
        with col2:
            st.metric("CEF Range", f"{cef_range:.6f}", help="Max - Min CEF values")
        with col3:
            st.metric("CEF Std Dev", f"{cef_std:.6f}", help="Standard deviation")
        with col4:
            st.metric("CEF Mean", f"{cef_mean:.6f}", help="Average CEF value")

        # CEF Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cycle_numbers,
            y=cef_values,
            mode='lines+markers',
            name='CEF',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))

        # Add trend line
        trend_y = model.predict(X)
        fig.add_trace(go.Scatter(
            x=cycle_numbers,
            y=trend_y,
            mode='lines',
            name=f'Trend (slope: {cef_slope:.6f})',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig.update_layout(
            title="CEF vs Cycle Number (First 10 Cycles)",
            xaxis_title="Cycle Number",
            yaxis_title="CEF Value",
            hovermode='x'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Additional visualizations
        st.subheader("📊 Additional Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Efficiency plot
            fig_eff = go.Figure()
            fig_eff.add_trace(go.Scatter(
                x=first_10_cycles['Cycle_Number'],
                y=first_10_cycles['Coulombic_Efficiency'],
                mode='lines+markers',
                name='Coulombic Efficiency',
                line=dict(color='green')
            ))
            fig_eff.add_trace(go.Scatter(
                x=first_10_cycles['Cycle_Number'],
                y=first_10_cycles['Energy_Efficiency'],
                mode='lines+markers',
                name='Energy Efficiency',
                line=dict(color='orange')
            ))
            fig_eff.update_layout(
                title="Efficiency Trends",
                xaxis_title="Cycle Number",
                yaxis_title="Efficiency"
            )
            st.plotly_chart(fig_eff, use_container_width=True)

        with col2:
            # Capacity plot
            fig_cap = go.Figure()
            fig_cap.add_trace(go.Scatter(
                x=first_10_cycles['Cycle_Number'],
                y=first_10_cycles['Charge_Capacity'],
                mode='lines+markers',
                name='Charge Capacity',
                line=dict(color='purple')
            ))
            fig_cap.add_trace(go.Scatter(
                x=first_10_cycles['Cycle_Number'],
                y=first_10_cycles['Discharge_Capacity'],
                mode='lines+markers',
                name='Discharge Capacity',
                line=dict(color='brown')
            ))
            fig_cap.update_layout(
                title="Capacity Trends",
                xaxis_title="Cycle Number",
                yaxis_title="Capacity (mAh)"
            )
            st.plotly_chart(fig_cap, use_container_width=True)

        # Download options
        st.subheader("💾 Download Results")

        # Create statistics DataFrame
        statistics_data = {
            'Parameter': ['CEF Slope (Linear Regression)', 'CEF Range', 'CEF Standard Deviation', 'CEF Mean'],
            'Value': [cef_slope, cef_range, cef_std, cef_mean],
            'Description': [
                'Linear regression slope of CEF vs Cycle Number for first 10 cycles',
                'Difference between maximum and minimum CEF values in first 10 cycles',
                'Sample standard deviation of CEF values for first 10 cycles',
                'Mean CEF value for first 10 cycles'
            ]
        }
        statistics_df = pd.DataFrame(statistics_data)

        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            statistics_df.to_excel(writer, sheet_name='CEF_Statistics', index=False)
            first_10_cycles.to_excel(writer, sheet_name='First_10_Cycles', index=False)
            final_dataset.to_excel(writer, sheet_name='Complete_Dataset', index=False)

        processed_data = output.getvalue()

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Complete Analysis (Excel)",
                data=processed_data,
                file_name="CEF_Analysis_Results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with col2:
            csv_data = final_dataset.to_csv(index=False)
            st.download_button(
                label="📄 Download Dataset (CSV)",
                data=csv_data,
                file_name="processed_battery_data.csv",
                mime="text/csv"
            )

        # Summary statistics table
        st.subheader("📋 CEF Statistics Summary")
        st.dataframe(statistics_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.write("Please ensure your Excel file has the correct format with columns: Time, Date, Voltage (mV), Current (mA), Capacity (mAh), Energy (mWh)")

else:
    st.info("👆 Please upload an Excel file to begin analysis")

    # Show sample data format
    st.subheader("📋 Required Data Format")
    sample_data = pd.DataFrame({
        'Sr. No.': [1, 2, 3, 4, 5],
        'Time': ['00:00:00.000', '00:00:30.000', '00:01:00.000', '00:01:30.000', '00:02:00.000'],
        'Date': ['2024-04-16 11:48:51.047'] * 5,
        'Voltage (mV)': [3708.14, 3723.11, 3730.12, 3735.12, 3740.11],
        'Current (mA)': [1274.12, 1274.11, 1274.12, 1274.11, 1274.11],
        'Capacity (mAh)': [0.18, 10.79, 21.41, 32.03, 42.64],
        'Energy (mWh)': [0.66, 40.12, 79.68, 119.30, 158.98]
    })
    st.dataframe(sample_data)


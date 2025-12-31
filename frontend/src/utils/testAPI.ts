// Simple API connection test
// Run this in browser console or as a separate test file

import { apiClient, checkAPIHealth } from '../services/api';

export const testAPIConnection = async () => {
  console.log('🧪 Testing Financial Seismograph API Connection...\n');

  try {
    // Test 1: Health Check
    console.log('1️⃣ Testing health check...');
    const isHealthy = await checkAPIHealth();
    console.log(`   Health Status: ${isHealthy ? '✅ Healthy' : '❌ Unhealthy'}`);

    if (!isHealthy) {
      console.log('❌ API is not available. Make sure backend is running on localhost:8000');
      return false;
    }

    // Test 2: System Stats
    console.log('\n2️⃣ Testing system stats...');
    const stats = await apiClient.getSystemStats();
    console.log(`   Articles processed: ${stats.articles_processed}`);
    console.log(`   Active feeds: ${stats.seismograph.active_feeds}`);
    console.log(`   Success rate: ${(stats.success_rate * 100).toFixed(1)}%`);

    // Test 3: Articles
    console.log('\n3️⃣ Testing articles endpoint...');
    const articles = await apiClient.getArticles();
    console.log(`   Retrieved ${articles.length} articles`);
    if (articles.length > 0) {
      console.log(`   Latest: "${articles[0].title.substring(0, 50)}..."`);
    }

    // Test 4: Seismograph Data
    console.log('\n4️⃣ Testing seismograph data...');
    const seismoData = await apiClient.getSeismographData();
    console.log(`   Data points: ${seismoData.length}`);

    // Test 5: Tremors
    console.log('\n5️⃣ Testing tremors...');
    const tremors = await apiClient.getTremors();
    console.log(`   Found ${tremors.length} tremors`);
    
    if (tremors.length > 0) {
      console.log(`   First tremor: "${tremors[0].title}"`);
      
      // Test 6: Epicenter Analysis
      console.log('\n6️⃣ Testing epicenter analysis...');
      try {
        const epicenter = await apiClient.getEpicenterAnalysis(tremors[0].id);
        console.log(`   Analysis retrieved for tremor: ${epicenter.tremor_id}`);
        console.log(`   Quality score: ${epicenter.quality_score}`);
      } catch (error: unknown) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        console.log(`   ⚠️ Epicenter analysis failed: ${errorMessage}`);
      }
    }

    // Test 7: AI Query
    console.log('\n7️⃣ Testing AI query...');
    try {
      const response = await apiClient.postQuery('What are the current market trends?', true);
      console.log(`   Query successful - response length: ${response.response?.length || 0}`);
      console.log(`   Processing times: ${JSON.stringify(response.processing_times)}`);
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.log(`   ⚠️ AI query failed: ${errorMessage}`);
    }

    console.log('\n✅ API connection test completed successfully!');
    console.log('\n🎯 Your dashboard should now show live data from the Financial Seismograph backend.');
    
    return true;

  } catch (error) {
    console.error('❌ API connection test failed:', error);
    console.log('\n🔧 Troubleshooting:');
    console.log('   1. Make sure your backend is running: python production_startup.py');
    console.log('   2. Check if localhost:8000 is accessible');
    console.log('   3. Verify no CORS issues in browser network tab');
    
    return false;
  }
};

// Export for use in components
export default testAPIConnection;